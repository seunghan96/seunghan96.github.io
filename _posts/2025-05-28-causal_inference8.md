---
title: Causal Inference - Part 8
categories: [ML, TS]
tags: []
excerpt: Causal Discovery - PC 알고리즘, FCI 알고리즘
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Causal Inference - Part 8

## Contents

1. PC 알고리즘
   1. 핵심 아이디어
   2. 주요 단계
   3. 의문점
   4. 장점 / 단점
   5. Python
2. FCI 알고리즘
   1. 핵심 아이디어
   2. PCI vs. FCI
   3. 주요 단계
   4. PAG (Partial Ancestral Graph)
   5. 장점 / 단점
   6. Python

<br>

# 1. PC 알고리즘 (Peter–Clark algorithm)

## (1) 핵심 아이디어

한 줄 요약: **조건부 독립성 검정**을 반복하여 **"변수 간의 인과 구조를 추론"**하는 대표적인 **Constraint-based causal discovery** 알고리즘

- 변수 간의 조건부 독립을 이용해 **불필요한 엣지를 제거**
- 남은 구조에 대해 규칙적으로 **방향성을 부여**하여 부분적으로 방향이 지정된 **DAG (PDAG)** 를 생성
- 모든 인과 관계가 **"관측된" 변수들로 설명된다**는 전제  (No latent confounders)

<br>

## (2) 주요 단계

| **단계**                                          | **설명**                                                     |
| ------------------------------------------------- | ------------------------------------------------------------ |
| **1. Fully-connected Graph 초기화**               | 모든 변수 쌍에 대해 edge가 있는 완전 무방향 그래프 생성      |
| **2. 조건부 독립성 검정으로 edge 제거**           | 점점 더 큰 조건 집합을 사용해 엣지를 제거, $Sepset(X,Y)$ 기록 |
| **3. 방향성 부여 (v-structure 및 DAG 규칙 적용)** | collider 구조 식별 후, 방향성 부여. 비순환성과 조건부 독립성 유지하도록 방향 확장 |

<br>

### Step 1: Initialize graph

- 변수 집합 $V = \{X_1, X_2, \dots, X_n\}$
- 초기 그래프 $G_0 = (V, E)$: Fully-connected + Undirected graph

<br>

### Step 2: 조건부 독립성 검정

모든 변수 쌍 $(X, Y)$에 대해, 

가능한 조건 변수 집합 $Z \subseteq V \setminus \{X, Y\}$ 에 대해, $X \perp Y \mid Z$ 이면 ...

$\rightarrow$ Edge $X - Y$ 제거

- 이때 $Z$는 $\text{Sepset}(X, Y)$ 로 저장

<br>

### Step 3: 방향성 부여

V-structure 판별 (collider)

- 만약 $X - Z - Y$, 그리고 $X \not\sim Y$ (Edge없음), 그리고 $Z \notin \text{Sepset}(X, Y)$ 이면

  => $X \rightarrow Z \leftarrow Y$

<br>

## (3) 의문점

```
PC 알고리즘에서 v-structure 판별 시에
왜 구조가 X → Z ← Y 처럼 Z로 방향이 향하는?
즉, 왜 Z가 양방향 수신점(collider)이 되는가?
Z에서도 다른 변수로 뻗어 나갈 수도 있지 않나?
```

대답: 관측된 **조건부 독립성** 패턴이 오직 **Z가 원인이 아니라 결과(collider)** 일 때만 논리적으로 설명이 가능하기 때문!

<br>

Details

- 그래프 구조: $X - Z - Y$
- X와 Y는 **엣지가 없음** → $X \perp Y$
- 그런데 X와 Y는 **Z를 조건으로 하면 의존함** → $X \not\perp Y \mid Z$

<br>

| **구조 형태** | **조건 없이**   | **Z로 조건 시**        | **방향**                   |
| ------------- | --------------- | ---------------------- | -------------------------- |
| $X ← Z → Y$   | $X \not\perp Y$ | $X \perp Y \mid Z$     | Z가 공통 원인              |
| $X → Z → Y$   | $X \not\perp Y$ | $X \perp Y \mid Z$     | Z가 중간 매개자            |
| $X → Z ← Y$   | $X \perp Y$     | $X \not\perp Y \mid Z$ | ✅ Z가 공통 결과 (collider) |

<br>

PC 알고리즘's 판단

1. X와 Y는 **엣지가 없음** → 독립

2. 그런데 X와 Y는 **Z를 조건으로 줄 때 의존하게 됨**

   ⟶ 이건 오직 Z가 **collider**일 때만 발생

<br>

## (4) 장점 / 단점

### a) 장점

- 직관적이며 이론적으로 강력
- 조건부 독립성 기반 → 해석력 있음
- 구현이 쉬움 (`causal-learn` 패키지 등) 

<br>

### b) 단점

- **숨은 변수**(latent confounder)에 민감 → 오류 발생
- 조건부 독립성 검정이 **샘플 수에 민감** 
- 조건 집합이 커질수록 계산량 급증 → 고차원 데이터에 비효율적 |

<br>

## (5) Python

```python
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

graph = pc(data, alpha=0.05, ci_test=fisherz, verbose=True)
```

<br>

# 2. FCI (Fast Causal Inference) 알고리즘

## (1) 핵심 아이디어

- PC 알고리즘을 확장한 방식
- How? **숨은 변수(latent variables)** 와 **선택 편향(selection bias)** 이 존재할 수 있는 현실적 환경에서 **관측된 변수들**만으로 **"부분적"** 인과 구조를 추론

<br>

## (2) PC vs. FCI

| **상황**              | **PC 알고리즘** | **FCI 알고리즘**                  |
| --------------------- | --------------- | --------------------------------- |
| 모든 원인이 관측됨    | 가능            | 가능                              |
| 숨은 변수 있음을 가정 | X               | O                                 |
| 선택 편향 존재        | X               | O                                 |
| 추론 대상             | 완전 DAG        | **PAG (Partial Ancestral Graph)** |

PC와의 공통점

- 조건부 독립성 기반으로 엣지를 제거함

PC와의 차이점

- **숨은 변수로 인한 v-structure 왜곡 가능성**을 고려해서, **추가적인 조건 집합(Possible-D-Sep)**을 활용
- 완전한 방향을 부여하는 대신, **불확실성을 포함한 방향**을 가진 **PAG** 

<br>

## (3) 주요 단계

PC 알고리즘의 Step 1~3은 동일하지만,

### Step 4) Possible-D-Sep

- (PC) 조건 집합 $Z$를 **"직접 연결"된 이웃만** 고려

- (FCI) **"간접적"으로 연결된 노드도 조건 집합으로 고려**

  → 숨은 변수에 의한 간접 경로를 제거할 수 있음

<br>

### Step 5) 추가 조건부 독립성 검정 및 방향성 확장

- Possible-D-Sep을 활용
- 확정 가능한 방향성만 부여하고, **불확실한 관계는 방향 없이 남김**

<br>

## (4) Output: **PAG (Partial Ancestral Graph)**

- 방향이 확실한 엣지: $X \rightarrow Y$
- 방향이 불확실하거나 숨은 변수 가능성 있음:
  - $X$ o→ $Y$
  - $X$ o–o $Y$ (latent confounder 가능성)

<br>

## (5) 장점 / 단점

(PC 알고리즘과 대비한 상대적 장/단점)

### a) 장점

- 숨은 변수까지도 고려! (보다 현실적)

### b) 단점

- 계산량 많음 (Possible-D-Sep 계산)
- 결과 해석이 더 복잡 (PAG 구조)

<br>

## (6) Python

```python
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz

graph = fci(data_matrix, alpha=0.05, ci_test=fisherz, verbose=True)
graph.draw()
```

