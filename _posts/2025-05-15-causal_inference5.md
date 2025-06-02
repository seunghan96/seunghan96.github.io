---
title: Causal Inference - Part 5
categories: [ML, CI]
tags: []
excerpt: Propensity Score Matching (PSM), Inverse Propensity Weighting (IPW), Doubly Robust (DR) Estimator
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Causal Inference - Part 5

## Contents

1. Backdoor path 끊기
1. Propensity Score Matching (PSM)
1. Inverse Propensity Weighting (IPW)
1. Doubly Robust (DR) Estimator

<br>

# 1. Backdoor path 끊기

CI에서 **confounding** 문제를 해결하기!

How? 처치(treatment)와 결과(outcome) 사이의 **backdoor path**를 끊기!

<br>

대표적인 방법론: Propensity Score 기반 기법

- (1) **Propensity Score Matching (PSM)**
- (2) **Inverse Propensity Weighting (IPW)**
- (3) **Doubly Robust (DR) Estimator**

<br>

# 2. Propensity Score Matching (PSM)

## (1) Propensity Score란

- 정의: 각 단위(예: 사람, 회사 등)가 처치를 받을 **확률**
- 수식: $$e(X) = P(T=1 \mid X)$$.
  - 즉,  **공변량 X**를 사용하여 처치 $$T \in \{0,1\}$$을 받을 확률을 추정!

<br>

## (2) 필요성

- RCM에서 **Ignorability**가 성립하려면 $$(Y(0), Y(1)) \perp T \mid X$$

  $$\rightarrow$$ But 현실에서 $$X$$가 고차원이면 **"조건부 독립을 만족시키기 어려움"**

- **Rosenbaum & Rubin (1983)**의 증명:

  - $$(Y(0), Y(1)) \perp T \mid e(X)$$
  - i.e., ***Propensity score만 맞추면 효과 비교 가능!***

<br>

## (3) PSM의 절차

Step 1) **Propensity Score 추정**

- $$e(X_i) = P(T_i = 1 \mid X_i)$$
- e.g., Random Forest, Logistic Regression, XGBoost

Step 2) **Matching**

- 각 treated unit을 비슷한 e(X) 값을 가진 control unit과 짝짓기 (nearest neighbor, caliper 등)

Step 3)  **인과 효과 추정**

- 매칭된 샘플에서 결과 평균 차이 계산:
  - $$\text{ATT} = \frac{1}{N_T} \sum_{i \in \text{treated}} (Y_i - Y_{\text{matched}(i)})$$.

<br>

## (4) Example

| **ID** | **X (나이)** | **T (처치)** | **Y (결과)** | **e(X)** |
| ------ | ------------ | ------------ | ------------ | -------- |
| A      | 25           | 1            | 85           | 0.80     |
| B      | 24           | 0            | 78           | 0.78     |
| C      | 45           | 1            | 70           | 0.30     |
| D      | 46           | 0            | 65           | 0.32     |

- $$A ↔ B, C ↔ D$$로 매칭
- 인과 효과 추정: $$\text{ATT} = \frac{(85 - 78) + (70 - 65)}{2} = \frac{12}{2} = 6$$.

<br>

# 3. Inverse Propensity Weighting (IPW)

## (1) PSM vs. IPW

- PSM = "개별 단위들"을 매칭
- IPW = 관측된 데이터를 **"가중(weighting)"**하여 마치 **"무작위 실험"**처럼 보이게 만듬

<br>

## (2) Key Idea

가중치 부여

- 처치받은 단위의 weight: $$\frac{1}{e(X)}$$
- 처치받지 않은 단위의 weight: $$\frac{1}{1 - e(X)}$$

<br>

$$\text{ATE} = \frac{1}{N} \sum_{i=1}^{N} \left[ \frac{T_i Y_i}{e(X_i)} - \frac{(1 - T_i) Y_i}{1 - e(X_i)} \right]$$.

<br>

## (3) Intuition

- 어떤 사람이 원래 80% 확률로 처치를 받을 사람이었는데, 실제로 처치를 받았다면,

  → “예상된 결과이므로 덜 중요한 정보” → 가중치 $$\frac{1}{0.8} = 1.25$$

- 반대로 20% 확률인데도 처치를 받았다면,

  → “희귀한 사례이므로 중요한 정보” → 가중치 $$\frac{1}{0.2} = 5$$

<br>

Result: 데이터의 **selection bias**를 보정!!

<br>

# 4. Doubly Robust (DR) Estimator

## (1) 개념

- IPW & Outcome Regression(결과 예측)을 동시에 사용

  - **두 가지 중 하나만 제대로 되면 ATE를 일관되게 추정 가능**

    (그래서 Doubly Robust)

<br>

## (2) 수식

$$\widehat{ATE}{DR} = \frac{1}{N} \sum{i=1}^N \left[ \left( \frac{T_i - e(X_i)}{e(X_i)(1 - e(X_i))} \right)(Y_i - \hat{Y}(X_i)) + \hat{Y}(1, X_i) - \hat{Y}(0, X_i) \right]$$.

- $$\hat{Y}(1, X), \hat{Y}(0, X)$$: outcome model (예: 회귀)
- $$e(X)$$: propensity score model

<br>

## (3) "Doubly" robust

- Propensity Score Model이 틀려도 Outcome Model이 맞으면 OK
- Outcome Model이 틀려도 Propensity Score Model이 맞으면 OK
- 둘 다 맞으면 더 정확!

<br>

# 5. Summary

| **방법** | **핵심 아이디어**                    | **장점**              | **단점**                    |
| -------- | ------------------------------------ | --------------------- | --------------------------- |
| PSM      | 비슷한 확률의 treated/control 짝짓기 | 직관적, 해석 쉬움     | 정보 손실 가능              |
| IPW      | 확률 역수를 가중치로 적용            | 전체 데이터 사용 가능 | 극단적인 확률에 민감        |
| DR       | IPW + Outcome 모델 결합              | 하나만 맞아도 일관성  | 구현 복잡, 추정 불안정 가능 |
