---
title: Causal Inference - Part 6
categories: [ML, CI]
tags: []
excerpt: Causal Inference in ML
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Causal Inference - Part 6

### Contents

1. ML vs. CI
2. ML에서 CI의 필요성
3. ML in CI
   1. CI 기반의 ML
   2. ML을 활용한 CI 보조
   3. DL 기반 CI 모델

<br>

# 1. ML vs. CI

| **항목**  | 머신 러닝 (ML)                         | **인과 추론** (CI)                              |
| --------- | -------------------------------------- | ----------------------------------------------- |
| 목표      | 예측 (예: $$P(Y \mid X)$$)             | 인과 효과 추정 (예: $$P(Y \mid \text{do}(X))$$) |
| 학습 방식 | 주어진 데이터로 패턴 학습              | 개입 또는 반사실 시나리오 고려                  |
| 전제      | i.i.d. (독립 동일 분포), 상관관계 중심 | 인과 구조, 개입 및 반사실 기반                  |
| 한계      | 개입 상황에서 잘 작동 안 함            | 개입/정책/설명에 강함                           |

<br>

# 2. ML에서 CI의 필요성

1. **일반화 (Generalization)**:
   - 데이터 분포가 바뀌면 기존 ML모델은 성능 저하 (e.g., covariate shift, concept drift)
   - 인과 모델은 **도메인 간 전이 학습(domain adaptation)**에 강함
2. **정책 결정 / 개입 추천**:
   - 다양한 분야에서  ***“무엇을 하면 결과가 바뀔까?”***라는 질문에 답하려면 인과 추론이 필요!
3. **설명가능성 (Explainability)**:
   - 상관관계가 아닌 **"원인-결과" 관계**를 알려줌

<br>

# 3. ML in CI

## (1) CI 기반의 ML

(SCM 또는 RCM 기반) CI 구조 + ML 모델

- e.g., 인과 그래프 (CI)를 학습 (ML)
- e.g., 인과 관계 (CI) 를 반영한 예측모델 (ML) 설계

<br>

Example

$$Y = f(X, \text{do}(T))$$.

- $$T$$: 개입 변수, $$X$$: 공변량, $$Y$$: 결과

$$\rightarrow$$ 인과 그래프를 이용해 backdoor adjustment로 보정

<br>

## (2) ML을 활용한 CI 보조

머신러닝 기법을 이용해 **인과 추론의 각 단계**를 자동화하거나 강화함:

| **단계**       | **인과 추론 작업**  | **머신러닝 활용 예**                        |
| -------------- | ------------------- | ------------------------------------------- |
| 인과 구조 학습 | DAG 구조 학습       | GNN, NOTEARS 등                             |
| 반사실 추정    | $$Y(0), Y(1)$$ 추정 | TARNet, CFRNet, GANITE 등                   |
| 효과 추정      | ATE, ITE 등         | X-Learner, T-Learner, S-Learner 등          |
| 변수 선택      | Confounder 탐지     | Representation learning, Invariant learning |

<br>

## (3) DL 기반 CI 모델

### a) TARNet / CFRNet

- 목적: 조건부 인과 효과 **(CATE: Conditional Average Treatment Effect)** 추정
- 핵심: 처치군과 대조군을 **shared representaiton space로 매핑**

<br>

### b) GANITE

- 목적: 개별 인과 효과 **(ITE: Individual Treatment Effect)** 추정 
- 핵심: GAN 기반으로 잠재 결과 Y(0), Y(1) 생성

<br>

### c) CausalBERT

- Text-based 인과 추론
  - BERT를 fine-tune하여 treatment, outcome 관계 파악

<br>

자세한건 다음 포스트에!
