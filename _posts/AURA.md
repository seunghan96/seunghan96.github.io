# AURA (Activation Unlearning with Ranking Activation) (ICML 2024)

📄 [논문: Whispering Experts: Neural Interventions for Toxicity Mitigation in Language Models](https://openreview.net/pdf?id=2P6GVfSrfZ)

<br>

## (1) 아이디어

- Background: LM 내부에는 어떤 neuron들이 특정 속성(e.g., toxic)을 강하게 담당하는 경우가 있음

- Proposal: **AURA**
  - neuron별로 “toxic 문장을 잘 구분하는 정도”를 측정
  - **toxic 관련 neuron을 억제(dampening)** 하여 출력을 조정

<br>

요약: **Neuron 레벨에서 활성화 기여도를 분석 **

→ 유해한 방향을 약화시켜서 모델이 그 속성을 덜 표현하도록 만듬

<br>

## (2) 과정

1. **Ranking activation**
   - 데이터셋 (e.g., toxic vs non-toxic 문장)을 준비
   - 각 **neuron의 activation 분포**를 분석하여, **toxic vs non-toxic을 잘 구분하는 neuron을 랭킹**
2. **Unlearning (dampening)**
   - toxic neuron들의 activation 값을 줄이거나 제거
   - non-toxic neuron은 그대로 유지
3. **Inference-time 적용**
   - 모델 추론 시점에 toxic neuron들이 강하게 활성화되는 걸 억제 → 생성 결과에서 유해성 감소

![figure2](/assets/img/llm/img829.png)

<br>

## (3) **장점**

- 직관적: ***“toxic neuron만 눌러버리자”***
- **fine-tuning 없이도** inference-time intervention 가능

<br>

## (4) 단점

- 범용성 X: 특정 task(e.g., toxic 억제)에는 효과 있지만, 다른 속성 제어에는 잘 작동하지 않음
- neuron 단위 제어는 세밀하지만, 의미적 해석이 어렵고 과도하게 억제하면 성능 저하

<br>

## (5) Details: neuron 랭킹

1. **데이터 수집**

- 문장 단위 레이블 $$y \in \{0\text{(non-toxic)},1\text{(toxic)}\}$$
- 모델을 한/여러 layer까지 통과시켜 **neuron 활성값** $$h \in \mathbb{R}^{B\times T\times D}$$ 수집
  - $$B$$: 배치, $$T$$: 토큰 길이, $$D$$: 히든 차원(=neuron 수)

<br>

2. **Sequence 수준으로 풀링**

- 문장 레이블이므로 토큰 축($$T$$)을 평균/최대 등으로 **풀링**해 $$h^{pool}\in \mathbb{R}^{B\times D}$$로 만듬
  - 예: mean-pooling, max-pooling, [CLS] 위치만 사용 등

<br>

3. **neuron별 분리도 점수 계산**

- 각 neuron $$j$$에 대해 $$h^{pool}_{:,j}$$ (길이 B)와 레이블 y 사이의 **분리도(separability)**를 측정해서 점수화.
- 대표적인 지표:
  - **AUC-ROC**(권장): 단일 특성(neuron)으로 toxic을 얼마나 잘 구분?
  - **Cohen’s d / t-통계량**: 클래스 간 평균 차이의 표준화 크기
  - **Point-biserial 상관계수**(연속 vs 이진)
  - **AUCPR**(양성 드문 경우)

<br>

4. **Normalization & layer 결합(선택)**

- layer마다 스케일이 다르면, **z-score normalization** 후 점수를 비교.
- 여러 layer를 쓸 거면 상위 $$k\%$$ neuron을 합치거나, layer별 상위 k를 골라 union.

<br>

5. **랭킹 & 억제(dampening)**

- 점수 내림차순으로 정렬 → 상위 M개를 **toxic neuron**으로 간주
- 추론 시 해당 neuron들 activation을 $$\alpha(0\!\sim\!1)$$ 배로 줄이거나(soft dampening), 바이어스 보정

