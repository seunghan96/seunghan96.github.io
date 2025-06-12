---
title: Active Learning (1) Uncertainty Sampling
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

# 3. Uncertainty-based Methods

## 기본 아이디어

모델이 예측에 **가장 확신이 없는 샘플**을 선택

<br>

## (1) Least Confidence (최소 신뢰도)

한 줄 요약: **가장 높은 softmax 확률값**이 낮은 샘플 선택

- 수식: $$LC(x) = 1 - \max_{c} P(y=c \mid x)$$
- 예시:
  - Sample A: [0.95, 0.03, 0.02] → LC = 0.05
  - Sample B: [0.45, 0.35, 0.20] → LC = 0.55 → B 선택

- 장/단점
  - 장점: 간단 + 직관적
  - 단점: 2번째 확신 수준 고려 X

<br>

## (2) Margin Sampling

한 줄 요약: **가장 확신 높은 두 클래스 간의 예측 차이**로 불확실성 측정

- 수식: $$\text{Margin}(x) = P(y=c_1) - P(y=c_2) \quad (c_1, c_2: \text{top-2 classes})$$
- 예시:
  - Sample A: [0.8, 0.15, 0.05] → margin = 0.65
  - Sample B: [0.41, 0.39, 0.2] → margin = 0.02 → B 선택

- 장/단점
  - 장점: 경쟁 클래스 (1등 vs. 2등) 반영
  - 단점: 전체 분포 고려 X

<br>

## (3) Entropy Sampling

한 줄 요약: 예측 분포 **전체의 불확실성** 측정

- 수식: $$H(x) = -\sum_{c} P(y=c \mid x) \cdot \log P(y=c \mid x)$$
- 예시:
  - Sample A: [1.0, 0.0, 0.0] → H = 0 (확신)
  - Sample B: [0.34, 0.33, 0.33] → H ≈ 1.58 (최대 불확실성)

- 장/단점
  - 장점: 전체 분포 고려
  - 단점: softmax 확률이 실제 확신을 반영하지 않을 수 있음

<br>

## (4) Variation Ratios 

한 줄 요약: 여러 번 예측 (e.g., dropout) 후 **가장 자주 예측된 클래스의 비율**로 불확실성 측정

- 수식: $$VR(x) = 1 - \frac{\text{mode count}}{T}$$
  - $$T$$ = sampling 횟수

- 예시: 
  - 모델이 샘플 x를 10번 예측 → 6번 클래스 A, 4번 클래스 B → VR = 0.4

<br>

# 3. 코드

위의 2-(3) Entropy Sampling 코드 예시

```python
import torch
import torch.nn.functional as F

def compute_entropy(logits):
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy

# logits: [batch_size, num_classes]
logits = model(unlabeled_batch)
entropy_scores = compute_entropy(logits)

# Top-K 불확실한 샘플 선택
topk_indices = torch.topk(entropy_scores, k=K).indices
selected_samples = unlabeled_batch[topk_indices]
```
