---
title: Causal Inference - Part 7
categories: [ML, TS]
tags: []
excerpt: Causal Discovery란
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Causal Inference - Part 7

### Contents

1. Causal Discovery란
1. Causal Discovery vs. Causal Inference
1. Causal Discovery의 대표적인 알고리즘

<br>

# 1. Causal Discovery란

**Causal Discovery (인과 구조 발견)**

- 인과 추론(causal inference)의 전단계
- 한 줄 요약: 관측된 데이터를 바탕으로 **변수들 간의 인과 구조** 를 추론

<br>

**Causal Inference**를 위해선...

- 인과 효과를 추정하려면 **먼저 인과 그래프(DAG)** 가 필요!
- → 이걸 자동으로 찾는 게 Causal Discovery

<br>

**Causal Discovery의 input & output**

|        | **내용**                                                     |
| ------ | ------------------------------------------------------------ |
| Input  | 관측 데이터                                                  |
| Output | 인과 그래프 (DAG 또는 PAG) = 변수 간 방향성 있는 네트워크 구조 |

<br>

**Causal Discovery의 필요성**

- 현실 데이터에는 변수 간 상관은 많지만, ***인과관계는 드러나지 않음***

<br>

# 2. Causal Discovery vs. Causal Inference

| **항목**           | **Causal Discovery**           | **Causal Inference**                       |
| ------------------ | ------------------------------ | ------------------------------------------ |
| 목적               | "인과 구조" 찾기               | (주어진 인과 구조 하에서) "인과 효과" 추정 |
| 입력               | 데이터                         | 데이터 + 구조                              |
| 질문 예시          | “누가 원인이고 누가 결과인가?” | “X가 Y에 얼마나 영향을 주는가?”            |
| 사용 알고리즘 예시 | PC, FCI, GES, NOTEARS, LiNGAM  | Backdoor, Propensity Score, IV 등          |

<br>

# 3. Causal Discovery의 대표적인 알고리즘

## (1) Constraint-based 계열 

- 변수 간 ***"조건부 독립성"*** 관계를 기반으로 인과 구조 유도
- 예: **PC**, **FCI**, **RFCI**

<br>

## (2) Score-based 계열

- 가능한 DAG 구조들 중에서 **BIC, AIC 등 점수**가 가장 좋은 그래프 선택
- 예: **GES**, **GIES**

<br>

## (3) Functional model 계열 

- **변수 간 함수 형태(선형/비선형)**를 가정해 방향성 추론
- 예: **NOTEARS**, **LINGAM**, **ANM**



