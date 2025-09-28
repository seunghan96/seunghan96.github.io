---
title: (Quantization) (6) QAT, PTQ
categories: [LLM, MULT]
tags: []
excerpt: QAT (Quantization-Aware Training), PTQ (Post-Training Quantization)
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Contents

1. Calibarion의 필요성
2. PTQ (Post-Training Quantization)
3. QAT (Quantization-Aware Training)
4. Example) INT8 & Symmetric
5. PTQ vs. QAT

<br>

# 1. Calibarion의 필요성

Quantization의 mapping (symmetric/asymmetric)

- $$q=\mathrm{clamp}\!\Big(\mathrm{round}\!\big(\tfrac{x}{s}+z\big),\,Q_{\min},\,Q_{\max}\Big)$$.
- $$\hat{x}=s\,(q-z)$$.

<br>

Notation

- s: **scale(스텝 크기)**
- z: **zero-point(오프셋)**
- $$Q_{\min},Q_{\max}$$: 정수 포맷 범위
  - e.g., 예: INT8 대칭이면 -127,127; 비대칭이면 0,255

<br>

**범위 초과** 시? ( == $$\tfrac{x}{s}+z$$가 정수 범위 밖으로 나가면 )

$$\rightarrow$$ Clipping!

<br>

### Calibration

$$s,z$$**를 데이터 분포에 맞게 정하는 과정**

<br>

대표적 선택 기준

(Clipping 임계값 T를 정하고 그로부터 s,z를 산출):

- **Min–Max**: $$T_{\min}=\min(x),\;T_{\max}=\max(x)$$.

- **Percentile**: (예) 상하위 0.1% 잘라 $$T_{\min},T_{\max}$$ 설정 → **아웃라이어 무시**

- **오차 최소화(MSE/MAE)**: $$\sum (x-\hat{x})^2$$ 최소가 되는 T 탐색

- **KL-divergence**: 원본 히스토그램 vs. 양자화 히스토그램 KL 최소

<br>

그 후

- **대칭(symmetric)**: 
  - $$z=0,\; s=T/Q_{\max}$$ 
    - with $$T=\max(\mid T_{\min}\mid ,\mid T_{\max}\mid )$$
- **비대칭(asymmetric)**: 
  - $$s=(T_{\max}-T_{\min})/(Q_{\max}-Q_{\min})$$.
  - $$z=\big\lfloor -T_{\min}/s \big\rceil$$.

<br>

# 2. PTQ (Post-Training Quantization)

학습 끝난 모델을 **retraining 없이** 양자화

1. **Calibration 데이터**(수십~수천 샘플)로 각 Tensor 분포 수집
2. $$T$$ 결정 → $$s,z$$ 계산
   - feat. 위 기준(Percentile/KL/MSE 등)
3. Weight/Activation Quantization
   - Weight는 보통 **per-channel**
   - Activation은 보통 **per-tensor**
4. 선택적으로 **바이어스 보정**, **레이어 스케일 평활(CLE/SmoothQuant 등)**

<br>

장/단점

- 장점: 빠름, 라벨 불필요
- 단점: 범위 결정이 부정확하면 **클리핑/해상도 손실**로 정확도 하락

<br>

LLM 특화 PTQ: **GPTQ, AWQ, NF4(Q-LoRA)** 등은 “가중치-전용” PTQ로 오차를 더 정교히 최소화

<br>

# **3) QAT (Quantization-Aware Training)**

Training/Fine-tuning 중 **가짜 양자화(FakeQuant)**를 넣어 NN가 Clipping에 **적응**하게 함.

- Forward: 위 식과 같은 round+clamp로 $$\hat{x}$$ 사용
- Backward: round에 **STE(Straight-Through Estimator)** 적용

<br>

Variants

- 스케일을 **학습 변수**로 두기도 함(예: **LSQ**)
- 활성화 클리핑 임계값을 **학습**(예: **PACT**)

<br>

장/단점

- 장점: FP와 근접한 정확도, **범위 초과에 강함**

- 단점: 데이터/시간 필요

<br>

# 4. Example) INT8 & Symmetric

- 데이터: $$\{-2,-1,0,1,\mathbf{8}\}$$

- 대칭 INT8: $$Q_{\max}=127$$

<br>

### **(a) Min–Max 기반 (아웃라이어 포함)**

임계값 

- $$T=\max \mid x\mid =8$$.
- $$s=T/127=8/127\approx 0.0630$$.

<br>

양자화/복원:

- $$-2 \Rightarrow q\approx -32,\; \hat{x}\approx -2.016$$.

- $$-1 \Rightarrow q\approx -16,\; \hat{x}\approx -1.008$$.

- $$1 \Rightarrow q\approx 16,\;  \hat{x}\approx 1.008$$.

- $$8 \Rightarrow q=127,\; \hat{x}\approx 8.001$$.

  → **아웃라이어는 정확**, 대신 스텝이 커져 일반 값 해상도가 거칠어짐.

<br>

### **(b) Percentile/MSE 기반(아웃라이어 무시,** T=3**로 클리핑)**

- $$s=3/127\approx 0.0236$$.
- 양자화/복원:
  - $$-2 \Rightarrow q\approx -85,\; \hat{x}\approx -2.008$$.
  - $$-1 \Rightarrow q\approx -42,\; \hat{x}\approx -0.992$$.
  - $$1 \Rightarrow q\approx 42,\;  \hat{x}\approx 0.992$$.
  - $$\mathbf{8} \Rightarrow \mathbf{q=127\;(clipping)},\; \hat{x}=\mathbf{3.0}$$.

<br>

# 5. PTQ vs. QAT

**PTQ**: 빠르게 배포, 라벨/재학습 어려울 때.

- Weight: **per-channel** Calibration
- Activation: **percentile/KL** Calibration 
- **첫/마지막 레이어 FP16 유지**

<br>

**QAT**: 정확도 민감, 4-bit/저비트, 배포 품질 중요할 때.

- **LSQ/PACT**로 스케일·클리핑 학습
- e.g., **지연 스케일 업데이트(EMA)**, **저비트 레이어만 QAT**
