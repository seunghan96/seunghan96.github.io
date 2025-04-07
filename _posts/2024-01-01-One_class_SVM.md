---
title: One-class SVM, SVDD
categories: [ML]
tags: []
excerpt: 

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# One-class SVM

## (1) 핵심 아이디어 요약

- (1) Unsupervised AD 모델 (**정상 데이터만을 이용해** 학습)
- (2) Procedure
  - Step 1) 데이터를 **고차원 feature space**로 매핑
  - Step 2) 그 공간에서 **원점으로부터 최대한 떨어진 초평면(hyperplane)**을 찾음
  - Step 3) 초평면 내부에 있으면 **정상**, 외부에 있으면 **이상(anomaly)**

<br>

## (2) 최적화

$$\min _{w, \rho_1(\xi)} \frac{1}{2} \mid \mid w \mid \mid ^2+\frac{1}{\nu n} \sum_{i=1}^n \xi_i-\rho$$

- subject to $$w \cdot \phi\left(x_i\right) \geq \rho-\xi_i, \quad \xi_i \geq 0, \quad i=1, \ldots, n$$

<br>

### Notation

- $$x_i$$: 입력 샘플
- $$\phi(x)$$: 커널을 통해 사상된 고차원 feature
- $$w$$: 초평면의 법선 벡터
- $$\rho$$: 경계 마진
- $$\xi_i$$: 슬랙 변수 (허용 오차)
- $$\nu$$: 하이퍼파라미터, **anomaly 허용 비율** (0~1 사이)

<br>

## (3) 해석

- 원점에서부터의 거리가 $$\rho$$ 이상인 feature vector $$\phi(x)$$들이 **정상**
- 그렇지 못한 점들은 슬랙 변수 $$\xi_i$$를 통해 **이상**으로 분류

<br>

## (4) RBF (Radial Basis Function) 커널

$$K\left(x, x^{\prime}\right)=\exp \left(-\frac{ \mid \mid x-x^{\prime} \mid \mid ^2}{2 \sigma^2}\right)$$.

<br>

## (5) Inference

테스트 데이터 $$x$$에 대해 결정 함수 $$f(x)$$를 다음처럼 정의

$$f(x)=\operatorname{sign}(w \cdot \phi(x)-\rho)$$.

- $$f(x) \geq 0$$: 정상
- $$f(x) < 0$$: 이상

<br>

## (6) 장점 & 단점

| 장점                        | 단점                                              |
| --------------------------- | ------------------------------------------------- |
| 정상 데이터만으로 학습 가능 | 고차원에서는 성능 저하 가능                       |
| 비선형 경계 지원 (커널)     | 하이퍼파라미터 조정 필요 ($$\nu$$, $$\sigma$$ 등) |
| 수학적 직관 명확            | 대규모 데이터에선 느림                            |

<br>

# SVDD (Support Vector Data Description)

## (1) 핵심 아이디어

- 마찬가지로, SVM 기반의 unsupervised AD 방법론 (**정상 데이터만** 가지고 학습)

- 데이터를 감싸는 **최소 구**를 찾는 이상 탐지 방법
  - feature space 상에서 **모든 정상 데이터를 감싸는 최소 반지름의 hypersphere**를 찾는다
- 테스트 샘플이 **구 내부**에 있으면 "정상", **외부**에 있으면 "이상"

<br>

## (2) SVDD vs One-Class SVM

| 항목      | SVDD                    | One-Class SVM                  |
| --------- | ----------------------- | ------------------------------ |
| 목적      | 최소 구로 데이터 감싸기 | 원점 기준으로 분리 초평면 찾기 |
| 직관      | 구 안이면 정상          | 초평면 위면 정상               |
| 수식 구조 | 중심-반지름 기반        | 초평면 margin 기반             |
| 성능      | 유사 (일반적으로)       | 유사                           |