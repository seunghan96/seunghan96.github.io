---
title: (Quantization) (7) FakeQuant, GGUF, GPTQ, AW, SpinQuant
categories: [LLM, MULT]
tags: []
excerpt: PTQ, QAT 방법론들
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Contents

1. QAT: FakeQuant
2. PTQ: GGUF (GPT Generated Unified Format)
3. PTQ: GPTQ (Post-Training Quantization for Transformers)
4. PTQ: AWQ (Activation-aware Weight Quantization)
5. QAT + PTQ: SpinQuant

<br>

# 1. QAT: FakeQuant

## (1) 개요

[한 줄 요약] 학습 중에 양자화 연산을 **흉내만 내는(faked)** 연산을 넣어, 네트워크가 양자화에 **적응하도록** 만드는 기법

- Forward에서는 “양자화했다가 다시 복원한 값”을 쓰고
- Backward에서는 **STE(Straight-Through Estimator)**로 미분을 흘려줌

$$\rightarrow$$ 학습이 끝나면, 실제 **정수 커널(INT8/INT4 등)로 교체**해 배포

<br>

## (2) Forward

Notation

- $$b$$: 비트 수
- 정수 범위 $$[Q_{\min}, Q_{\max}]$$ 
  - e.g., 대칭 INT8은 [-127,127]
  - e.g., 비대칭 INT8은 [0,255]
- $$x$$: FP32 텐서(가중치 또는 활성화)
- $$s$$: scale(스텝 크기)
- $$z$$: zero-point(오프셋, 대칭이면 보통 0)
- $$\hat{x}$$: 양자화-복원된 값(이 값이 다음 연산으로 전달)

<br>

**양자화 → 복원**(FakeQuant):

$$\begin{aligned} q &= \mathrm{clamp}\!\left(\mathrm{round}\!\left(\frac{x}{s} + z\right),\; Q_{\min},\; Q_{\max}\right),\\ \hat{x} &= s \cdot (q - z), \end{aligned}$$.

<br>



**대칭(symmetric)**: 

- $$z=0,\; s=\dfrac{T}{Q_{\max}}$$ with $$T=\max(\mid T_{\min}|,|T_{\max}|)$$

**비대칭(asymmetric)**: 

- $$s=\dfrac{T_{\max}-T_{\min}}{Q_{\max}-Q_{\min}},\; z=\left\lfloor-\dfrac{T_{\min}}{s}\right\rceil$$.

<br>

## (3) Backward, STE

round와 clamp는 **"미분 불가능"**이므로, QAT에서는 **STE**를 사용

- 아이디어: *전달할 그레이디언트는 1로 가정* (클리핑 구간 밖은 0)

- 직관적 표기:

  $$\frac{\partial \hat{x}}{\partial x} \approx \begin{cases} 1 & \text{(클리핑 범위 내부)}\\ 0 & \text{(클리핑 범위 바깥, saturation)} \end{cases}$$.

$$\rightarrow$$ 이렇게 해서 양자화가 넣는 불연속성에도 **학습이 진행**되도록!!

<br>

## (4) 어디에 붙이나? (삽입 위치)

(1) **가중치 (Conv/Linear weight)**: 

- 보통 **per-channel** 스케일(출력 채널별 scale)로 FakeQuant 삽입

(2) **활성화 (activation)**: 

- 보통 **per-tensor** 스케일, 연산 노드(Conv/Linear) **출력 쪽**에 FakeQuant 삽입

<br>

## (5) Scale/ Zero-point

1. **Observer(관찰자) 기반**:
   - MinMax, MovingAverageMinMax, Histogram, Percentile 등으로 $$T_{\min}, T_{\max}$$ 추정 → s,z 계산
   - QAT 시작 시 **칼리브레이션 몇 배치**로 초기화 후, 학습 동안 EMA로 갱신하거나 고정
2. **학습형(learnable) 스케일**:
   - **LSQ**류: s를 파라미터로 두고 학습 (역전파 시 STE 변형)
   - **PACT**류: 활성화 클리핑 임계값 $$\alpha$$를 **학습**
     - ReLU 앞뒤에 $$\mathrm{clip}(x; 0,\alpha)$$

<br>



Tip) 

- 4-bit 같은 **저비트**로 갈수록, **학습형 스케일/클리핑**이 정확도 유지에 유리!

- **초기**엔 FakeQuant를 껐다가 (Observer만 켜서 통계 수집) → **수 epoch 뒤** 켜기(학습 안정)
- 첫/마지막 레이어, 임베딩 레이어는 **고정밀(FP16/BF16)** 유지
- 가중치는 **per-channel**, 활성화는 **per-tensor**가 일반적
- BatchNorm은 QAT 전에 **Conv에 fuse**하는 게 보편적

<br>

## Pytorch

```python
import torch
import torch.nn as nn
from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver

# 예: 대칭 INT8, activation용 FakeQuant 모듈
act_fakequant = FakeQuantize(
    observer=MovingAverageMinMaxObserver,
    quant_min=-127, 
    quant_max=127,   # 대칭
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric,
    reduce_range=False
)

# 간단한 forward에서 사용
x = torch.randn(4, 16)  # activation
x_q = act_fakequant(x)  # forward에서 양자화-복원된 값으로 대체
```

<br>

## PTQ에 비해 효과적인 이유

- PTQ는 사후에 스케일을 **맞추기만** 하므로! 

  $$\rightarrow$$ clipping 손실을 **모델이 보상하지 못함**

- QAT는 학습 중 FakeQuant로 인한 손실을 **모델 parameter가 학습적으로 보상** 

  → 저비트(특히 4-bit)에서 효과 큼

<br>

# 2. **PTQ: GGUF (GPT Generated Unified Format)**

## (1) 개요

GGUF (GPT Generated Unified Format)

- *llama.cpp* 계열 툴체인에서 사용하는 **model 저장 포맷**
  - e.g., GGUF, safetensors, ckpt → 특정 규칙에 맞게 weight+메타데이터 저장
- LLM 모델을 **효율적으로 저장하고 추론**할 수 있도록 만든 **최신 PTQ(Post-Training Quantization) 포맷**
- QAT (X), PTQ (O)
  - 즉, QAT(학습 중 양자화)가 아니라, **이미 학습된 모델을 사후에 양자화**해 저장하는데 사용

<br>

**llama.cpp**란?

- Meta의 **LLaMA 모델**을 CPU나 작은 GPU에서도 돌릴 수 있게 만든 오픈소스 프로젝트
- 파이썬이나 거대한 GPU 환경이 없어도, 단순한 C/C++ 코드로 모델을 실행 가능
- 여기서 모델을 불러올 때는 **특정 파일 포맷**을 요구하는데, 그 최신 표준이 **GGUF**

<br>

## (2) 특징

- (1) **다양한 정밀도 지원**

  - FP16, INT8, INT4 등 여러 정밀도로 model 변환 가능

  - 특히 LLaMA, Mistral, Falcon 같은 model을 **4-bit, 8-bit로 PTQ** 해서 PC/모바일에서도 돌릴 수 있게 함.

- (2) **구조적 저장**

  - 단순한 weight 배열이 아니라, 
  - 메타데이터(토크나이저 정보, 레이어별 설정 등)까지 함께 담아 **완전한 모델 package** 

- (3) **효율성**

  - 기존 GGML 포맷보다 빠른 로딩, 더 작은 디스크 용량, 호환성↑

  - GPU 없이도 CPU에서 가볍게 구동 가능하도록 설계됨.

<br>

## (3) PTQ와의 연결

GGUF는 **양자화 알고리즘 자체**가 아님!

**양자화된 결과물을 담는 컨테이너** 역할임

- 예를 들어, LLaMA-7B를 INT4로 PTQ 했다면 → 그 결과물이 GGUF 파일(`model.Q4_K_M.gguf`)로 저장됨.
- 따라서 GGUF는 PTQ 기반 배포에서 **표준 포맷**처럼 사용

<br>

## (4) Summary

- **GGUF = PTQ된 모델을 담는 최신 통합 포맷**
  - GGUF는 **양자화 알고리즘 자체**가 아님!
  - **양자화된 결과물을 담는 컨테이너** 역할임

- e.g., LLaMA-7B를 INT4로 PTQ 했다면 

  → 그 결과물이 GGUF 파일(`model.Q4_K_M.gguf`)로 저장됨.

- INT8, INT4 PTQ 모델 배포에서 사실상 표준처럼 사용

<br>

좋습니다. 이제 **3) PTQ: GPTQ**에 대해 설명드릴게요.



------





# **3) PTQ: GPTQ (Post-Training Quantization for Transformers)**

## (1) 개요

**GPTQ** = “Post-Training Quantization for Transformers”

- (Note that 이름과 달리 단순히 GPT 모델만을 위한 건 아님)

  $$\rightarrow$$ **모든 Transformer 계열 LLM에 적용할 수 있는 PTQ 방식**

핵심: 

- 단순히 min–max 기반으로 스케일링 X
- 행렬 곱 연산에서 생기는 **quantizarion error를 최소화**하도록 weight를 양자화

<br>

## (2) 동작 방식

- (1) **Block 단위 처리**
  - Layer-wise → Block-wise
    - Transformer의 큰 weight 행렬(예: Linear layer)을 작은 블록(열 단위)으로 나눔.
  - 각 **"블록 별로 최적의 scale/zero-point"**를 찾음.

- (2) **Quantiaztion error 최소화**

  - 단순히 round 하는 것이 아님!
  - weight 행렬 $$W$$와 양자화된 $$W_q$$를 사용했을 때의 **출력 차이** $$\mid \mid  XW - XW_q \mid \mid $$ 를 최소화하도록 weight를 선택
    - 여기서 $$X$$는 sample 데이터(== calibration 데이터).

- (3) **순차 최적화(Sequential approximation)**

  - 한 열(column)을 양자화 → 그로 인한 오차를 고려 → 다음 열을 보정하면서 양자화.

  - 이렇게 하면 누적 오차를 줄일 수 있음.

<br>

## (3) Example

- 원래 Linear 레이어 weight: 4096 × 11008 행렬 (FP16)

- INT4로 단순 quantization → 성능 급락

- GPTQ 적용 시:

  

  - 샘플 입력 배치를 넣고, weight 열 단위로 최적 스케일 계산
  - 오차 최소화 → INT4에서도 원래 성능에 근접

  



- **GPTQ = 고도화된 PTQ 알고리즘**
- 핵심 = weight 행렬을 열 단위로 최적화해 **출력 오차 최소화**
- 효과 = INT4에서도 정확도 손실 최소화, LLM 배포에 많이 쓰임 (예: model.gptq 파일 형식으로 제공됨)



## (4) Summary

- **GPTQ = 고도화된 PTQ 알고리즘**
- 핵심 = weight 행렬을 열 단위로 최적화해 **출력 오차 최소화**

- **PTQ** 방식 → 추가 학습 (finetuning) 필요 없음.
- INT4에서도 정확도 손실 최소화, LLM 배포에 많이 쓰임 (예: model.gptq 파일 형식으로 제공됨)
- Details
  - **비트 수**: 주로 INT4, INT3에서도 좋은 성능.
  - **정확도**: 단순 uniform PTQ보다 훨씬 좋음.
  - **계산량**: 최적화 과정 때문에 변환 시간이 조금 더 걸림.

<br>

# 4. PTQ: AWQ (Activation-aware Weight Quantization)

## (1) 개요

**AWQ** (**Activation-aware Weight Quantization**)

- 단순히 weight 분포만 보고 양자화하지 않고, 

  **활성화(activation)와의 상호작용**을 고려하여 가중치를 양자화!

- PTQ 기법

- 목표: **INT4 양자화에서도 GPTQ 수준의 성능 유지**, 하지만 더 빠르고 간단하게!

<br>

## (2) 핵심 아이디어

- (1) **weight 중요도 측정**

  - 모든 weight가 모델 출력에 같은 영향을 주는 게 아님.

  - 특히 **activation이 크게 반응하는 방향**의 weight는 더 중요.

  $$\rightarrow$$ ***따라서 weight마다 “중요도”를 측정***

- (2) **중요한 weight 보존**

  - 중요도가 높은 weight: scale factor를 조정해서 **손실이 덜 나도록** 양자화.

  - 중요도가 낮은 weight: 더 거칠게 양자화해도 괜찮음

- (3) **결론**

  - 전체 오차(MSE)를 최소화하는 대신,
  - **모델 "출력에 더 큰 영향"을 주는 부분**을 정밀하게 보존.

<br>

## (3) Example

$$y = Wx$$.

- GPTQ: $$W$$의 양자화 오차를 최소화 ($$\mid \mid XW - XW_q \mid \mid$$)
- AWQ: $$Wx$$에서 activation $$x$$와 곱했을 때 영향 큰 weight를 더 잘 보존

<br>

즉, **입력(activation) 분포까지 고려해서 quantization을 최적화**한다는 차이가 있음.





## (4) Summary

- **PTQ** 방식 → 학습(finetuning) 불필요.
- **빠름**: GPTQ처럼 복잡한 최적화가 필요 없음 → 변환 속도 ↑.
- **효과적**: INT4에서도 FP16 성능에 근접.
- **구현**: Hugging Face awq 라이브러리 등에서 쉽게 사용 가능.

<br>

## (5) GPTQ vs. AWQ

### GPTQ

- **문제식**: $$\min_{W_q} \; \\mid  XW - XW_q \\mid ^2$$.
  - 여기서 X는 **칼리브레이션 데이터(activation batch)** 전체, W는 weight.
- 즉, **행렬 곱의 출력 차이 전체(norm) 최소화**가 목표.
- 구현상은 **열(column) 단위**로 순차 양자화 → 누적 오차를 줄이는 방식.

$$\rightarrow$$ **결과**: “모든 weight가 공평하게 고려됨.”

- 큰 activation 방향이 자동으로 더 큰 오차를 주기는 하지만,
- GPTQ는 weight를 직접적으로 중요/비중요로 나누지 않고, 그냥 전체 오차를 줄이는 최적화 문제로 접근.

<br>

### AWQ

- **문제식**: $$\min_{W_q} \; \\mid  X(W - W_q) \\mid ^2$$ 
  - with reweighting by activation importance
- 차이: 
  - X의 분포를 단순히 “샘플 데이터”로 보는 게 아니라,
  - **activation이 큰 축**(즉, 모델이 자주 크게 반응하는 weight 방향)을 강조해서 보존하도록 설계.
- 실무적으로는 **중요한 weight만 FP16 그대로 두고 나머지만 INT4**로 양자화하거나,
  - scale factor를 중요 weight에 유리하게 조정.

$$\rightarrow$$  **결과**: “활성화와 곱했을 때 영향력이 큰 weight는 더 정밀하게 보존, 덜 중요한 weight는 손실 감수.”

<br>

### Example

- 데이터: $$x = [1000, \; 1], weight W = [0.01, \; 0.01].$$
- 출력: $$y = 1000 \cdot 0.01 + 1 \cdot 0.01 = 10.01$$

<br>

GPTQ

- 두 weight 모두 0.01이니까 → 똑같이 양자화됨.
- 결과적으로 activation이 큰 첫 번째 weight가 약간 손실되면 → 출력 오차 커짐.
- GPTQ는 **출력 오차 전체**를 줄이려 애쓰지만, weight 중요도를 명시적으로 다루지 않음.

<br>

AWQ

- $$x_1 = 1000$$이 크므로 첫 weight(0.01)는 **매우 중요**.
- 따라서 이 weight를 FP16에 남겨두거나, 더 정밀하게 스케일링.
- 반대로 x_2=1 방향은 영향이 작으므로 INT4로 손실 감수해도 됨.
- 결과: 출력 오차는 훨씬 작아짐.

<br>

# 5. SpinQuant

## (1) 개요

**SpinQuant**

- 2024년에 제안된 **QAT + PTQ 기법**
- **특정 레이어를 회전(rotate) 변환**한 뒤 양자화를 적용
  - “Spin” = weight 공간을 **회전(spin)시켜 더 양자화 친화적인 space로 바꾸는 것**

<br>

## (2) 핵심 아이디어

- (1) **문제**:

  - LLM weight는 **"특정 축 방향으로 분산이 크거나 긴 꼬리 분포(outlier)"**가 존재.

  - 따라서, 그대로 (INT4/INT8로) 양자화하면 **일부 축에서 심한 손실 발생**

- (2) **해결**:

  - Step 1) weight를 **직교 행렬(orthogonal transform)**로 회전시켜 분포를 더 **“균등”하게 만든 뒤**에 양자화!

  - Step 2) 양자화 후 **다시 역회전(inverse transform)**하여 원래 space로 복원!

- (3) **학습(Adaptation)**:

  - 단순한 고정 회전 X
  - **저차원 어댑터(LoRA처럼 작은 학습 가능 모듈)**를 사용해 회전 행렬을 학습 O

  $$\rightarrow$$ 모델 전체를 다시 학습할 필요 없이, 작은 추가 학습만으로 양자화 친화적인 weight 공간 확보

<br>

## (3) Example

- 원래 weight 벡터: $$[10, 0.1, 0.05]$$ 

  - 첫 번째 값이 너무 큼 → 아웃라이어

- 그대로 INT4 quantization 시?

  → 첫 번째 값 때문에 scale이 커져, 나머지 값들(0.1, 0.05)이 전부 0으로 사라짐!

- SpinQuant: 

  - **weight 공간을 회전시켜 [2, 2, 2] 같은 균등 분포**로 만든 뒤 양자화 → 손실 최소화!!
  - 양자화된 상태에서 역회전 → 원래 의미를 잘 살린 weight 복원.

<br>

## (4) Summary

- **SpinQuant = weight 공간 회전(orthogonal transform) + 소규모 학습 → 양자화 친화적 표현으로 변환 후 양자화**
- **PTQ + 소규모 학습 = Hybrid**
  - PTQ처럼 전체를 재학습하지 않지만, GPTQ/AWQ보다 조금 더 학습을 허용.

- **저비트(특히 4-bit 이하)**에서 큰 성능 개선.
- 다양한 Transformer 구조(LLaMA, Mistral 등)에 적용 가능.

