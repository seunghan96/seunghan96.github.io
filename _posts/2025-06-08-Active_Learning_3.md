---
title: Active Learning (3) Core-set & Diversity-based
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

# 3. Core-set & Diversity-based

## 기본 아이디어

데이터 전체를 잘 대표하는 샘플(대표성, representativeness)과 **다양한 분포 영역에서 골고루** 선택된 샘플(다양성, diversity)을 선택!

i.e., ***“가장 혼란스러운 샘플”이 아니라, “라벨링을 통해 가장 많은 영역을 커버할 수 있는 샘플은 무엇인가?”***

<br>

## (1) k-Center Greedy Algorithm

- https://arxiv.org/pdf/1708.00489

```
Sener, Ozan, and Silvio Savarese. "Active learning for convolutional neural networks: A core-set approach." arXiv preprint arXiv:1708.00489 (2017).
```

<br>

핵심 아이디어: 현재 라벨된 데이터가 잘 **커버하지 못한 영역**을 우선 탐색

- Procedure

  - Step 1) **Feature 추출**

    - Unlabeled 데이터 전체에 대해 feature embedding을 계산

      (e.g., CNN의 마지막 pooling layer 또는 projection head를 사용)

    - $$f(x) \in \mathbb{R}^d$$.

  - Step 2) **초기 Core-set 구축**

    - 라벨된 데이터들의 feature $$\mathcal{L} = \{ f(x_i) \}$$

  - Step 3) **Distance 계산**

    - 각 unlabeled 샘플 u에 대해:

      $$d(u, \mathcal{L}) = \min_{l \in \mathcal{L}} \mid \mid f(u) - f(l) \mid \mid_2$$.

  - Step 4) **가장 먼 샘플 선택**
    - $$\max_{u \in \mathcal{U}} d(u, \mathcal{L})$$.
    - 선택된 샘플을 Core-set에 추가 & 반복
  - **K개 샘플 선택 완료 시 종료**

- 장/단점

  - 장점: 다양한 분포 영역을 고르게 커버

  - 단점: 
    - 모든 pairwise 거리 계산 필요 → 계산 비용 높음 $$\mathcal{O}(N \times k)$$
    - feature embedding의 품질에 의존적

![figure2](/assets/img/CONT/img42.png)

<br>

## (2) Clustering-based

- https://proceedings.mlr.press/v16/bodo11a/bodo11a.pdf

```
Bodó, Zalán, Zsolt Minier, and Lehel Csató. "Active learning with clustering." Active Learning and Experimental Design workshop In conjunction with AISTATS 2010. JMLR Workshop and Conference Proceedings, 2011.
```

<br>

핵심 아이디어: 데이터를 clustering & **각 cluster의 중심을 선택**

- Procedure

  - Step 1) 모든 unlabeled 데이터에 대해 **embedding 추출**

  - Step 2) **Clustering**

  - Step 3) For each cluster ...

    - Centroid에 가까운 샘플 1개 선택

      ( 혹은 entropy, uncertainty, 거리 기준으로 top-k 선택 )

- 기타: 각 cluster 크기 비례로 sample 수 조절

- 장/단점

  - 장점:

    - 라벨링 샘플의 **분포 균형 확보**

    - 소수 클래스 영역도 잘 탐색 가능

    - semi-supervised AL에 적합 (pseudo-label + cluster)

  - 단점: 

    - cluster 수 설정 어려움

<br>

# 4. 코드

k-Center Greedy Algorithm 예시

```python
import torch
import numpy as np

def compute_coreset_indices(embeddings_labeled, embeddings_unlabeled, k):
    selected = []
    dist = torch.cdist(embeddings_unlabeled, embeddings_labeled).min(dim=1).values
    for _ in range(k):
        idx = torch.argmax(dist)
        selected.append(idx.item())
        new_point = embeddings_unlabeled[idx].unsqueeze(0)
        dist = torch.minimum(dist, torch.cdist(embeddings_unlabeled, new_point).squeeze(1))
    return selected
```

