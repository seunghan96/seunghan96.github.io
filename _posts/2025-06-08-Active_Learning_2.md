---
title: Active Learning (2) Bayesian-based Uncertainty
categories: [CONT, CV, ML]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<br>

# Active Learning (AL)

# 1. Introduction

- 배경: 높은 Labeling 비용

- 아이디어: Unlabeled data 중, labeling을 할 데이터를 능동적으로 선택

- 주요 특징 및 개념

  - **Label Efficiency**: 모든 데이터를 라벨링하지 않고도 성능 향상 가능.

  - **Query Strategy**: 모델이 가장 정보가 될 만한 데이터를 “질문”해서 라벨을 요청.

  - **Iteration**: 모델 학습 → informative sample 선택 → 라벨 요청 → 학습 반복.

<br>

# 2. 주요 쿼리 전략 (Query Strategies)

1. **Uncertainty Sampling**
   - 모델이 가장 **불확실**해하는 샘플 선택 (e.g., 가장 낮은 softmax confidence)
2. **Query by Committee (QBC)**
   - 여러 모델로 구성된 committee 간 **예측 불일치가 큰** 샘플 선택
3. **Expected Model Change / Expected Error Reduction**
   - 어떤 샘플을 학습하면 성능이 얼마나 향상될지를 예측하여 선택
4. **Core-set Approaches**
   - 데이터 분포 전체를 대표할 수 있는 샘플 집합 선택

<br>

# 3. Bayesian-based Methods

## 기본 아이디어

DL 모델의 softmax 출력은 실제 **uncertainty(불확실성)**를 잘 표현하지 못하는 경우가 많음!

따라서, Bayesian 기반 접근을 통해, 예측값이 아닌 **모델 파라미터 자체의 분포**를 고려해 **예측 불확실성**을 추정!

<br>

## (1) Monte Carlo Dropout (MC-Dropout)

한 줄 요약: Dropout을 학습뿐만 아니라 **추론 시에도 활성화**시켜서 모델을 여러 번 샘플링

- 각 예측 결과의 분산(variance)을 불확실성으로 간주

- 수식: 
  - $$\mu = \frac{1}{T} \sum_{t=1}^{T} f(x;\theta_t)$$.
  - $$\text{Var}(f(x)) = \frac{1}{T} \sum_{t=1}^{T} (f(x;\theta_t) - \mu)^2$$.

- Procedure
  - Step 1) Dropout을 켜둔 채 inference 수행
  - Step 2) 각각의 softmax 확률 출력 저장
    - 평균 → 예측값
    - 분산 → 불확실성

- 장/단점

  - 장점: 

    - 기존 모델에서 dropout만 추가하면 사용 가능 (간단함)

    - 불확실성 정보가 실제 모델 weight 분산을 반영

  - 단점:

    - 추론 시간 **$$T$$배 증가**
    - dropout 비율 및 샘플링 수 하이퍼파라미터 민감

![figure2](/assets/img/CONT/img39.png)

<br>

## (2) Deep Ensembles

한 줄 요약: 서로 다른 초기화로 학습한 **복수의 모델**을 ensemble하여 예측 + 각 모델의 예측값 차이를 통해 **uncertainty 측정**

- 수식:
  - 예측 평균: $$\mu = \frac{1}{M} \sum_{m=1}^{M} f_m(x)$$
  - 예측 분산: $$\sigma^2 = \frac{1}{M} \sum_{m=1}^{M} (f_m(x) - \mu)^2$$

- Procedure:
  - Step 1) 동일한 네트워크 구조로 여러 모델 학습 (w/ different initialization)
  - Step 2) 각 모델로 inference → softmax 결과 평균/표준편차
    - 표준편차가 크면 → **모델 간 예측 불일치** → 높은 불확실성

- 장/단점

  - 장점: 
    - 다양한 관점의 예측이 반영됨 → robust하고 calibrated한 uncertainty
    - empirical하게 매우 우수한 성능 (e.g. OOD detection, AL 등)

  - 단점: 
    - **학습 비용 높음** (모델 M개 학습)

![figure2](/assets/img/CONT/img40.png)

<br>

## (3) Bayesian Neural Networks (BNNs)

한 줄 요약: 모델 파라미터 $$\theta$$를 **확률 변수**로 간주 + 예측 결과는 weight 분포로부터의 **샘플링 평균**

- Procedure
  - Step 1) Prior $$p(\theta)$$ 설정 (ex. Gaussian)
  - Step 2) Posterior $$p(\theta \mid D)$$ 근사 
  - Step 3) $$p(y \mid x) = \int p(y \mid x, \theta) p(\theta \mid D) d\theta$$

- 근사 방법:  **variational approximation** / **Laplace approximation**을 통해 tractable하게!
- 장/단점
  - 장점: 예측 불확실성+ 모델 불확실성 모두 고려
  - 단점: 최적화 수렴 어려움, 학습 시간 ↑

![figure2](/assets/img/CONT/img41.png)

<br>

# 4. 코드

MC-Dropout 기반 Uncertainty

```python
def enable_dropout(model):
    """Enable dropout layers during inference"""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

def predict_with_uncertainty(f_model, x, n_iter=30):
    enable_dropout(f_model)
    preds = torch.stack([F.softmax(f_model(x), dim=1) for _ in range(n_iter)])
    mean = preds.mean(dim=0)
    std = preds.std(dim=0)
    return mean, std
```



