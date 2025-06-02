---
title: Causal Inference - Part 4
categories: [ML, CI]
tags: []
excerpt: Structural Causal Model (SCM) 상세
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Causal Inference - Part 4

## Contents

1. SCM 개요
2. SCM의 3단계 추론
3. 개입: do-연산자
4. Backdoor Criterion (역인과 경로 차단)
5. RCM vs. SCM

<br>

# **1. SCM 개요**

## (1) 핵심 아이디어

핵심: **인과적 구조 (equations + graph)**

- 즉, 변수들 간의 관계를 **함수적 관계 (structural equations)**로 명시
- 이를 **인과 그래프(DAG)**로 나타냄

<br>

## (2) 구성 요소

| **구성 요소**                            | **설명**                                                     |
| ---------------------------------------- | ------------------------------------------------------------ |
| **변수 (Variables)**                     | 원인과 결과를 포함하는 모든 변수들 (관측/비관측 포함)        |
| **구조적 방정식 (Structural Equations)** | 각 변수는 부모 변수들의 함수로 표현됨. 예: $$Y = f(X, U_Y)$$ |
| **인과 그래프 (Causal DAG)**             | 변수 간 인과 관계를 나타내는 방향성 그래프                   |

<br>

## (3) Example

- 변수: $$X$$ (공부 시간), $$Y$$ (시험 점수)

- 비관측 변수: $$U_Y$$ (지능, 잠재적 능력)

- 구조적 방정식:

  $$Y = f(X, U_Y) = 2X + U_Y$$

- DAG:

```
X  ---->
         Y
U_Y ---->
```



<br>

# 2. SCM의 3단계 추론

**Pearl의 Ladder of Causation**

| **단계** | **종류**                | **질문 예시**                                             | **관련 도구**                      |
| -------- | ----------------------- | --------------------------------------------------------- | ---------------------------------- |
| 1단계    | 관찰 (Association)      | “시험 점수가 높은 사람들은 공부를 많이 하나요?”           | 조건부 확률 P(Y \mid X)            |
| 2단계    | 개입 (Intervention)     | “공부 시간을 강제로 5시간으로 정하면 점수는 어떻게 될까?” | 개입 확률 P(Y \mid \text{do}(X=5)) |
| 3단계    | 반사실 (Counterfactual) | “그 학생이 공부를 했더라면 더 좋은 점수를 받았을까?”      | P(Y_{x’} \mid X = x, Y = y)        |

<br>

# 3. 개입: do-연산자

개입 = ***강제로 조작***

수학적 표현: $$P(Y \mid \text{do}(X=5))$$

- 이는 단순한 조건부 확률 $$P(Y \mid X=5)$$ 와는 다름!
- (1) 조건부 확률: “공부 시간 **관찰**값이 5시간인 학생의 평균 점수”
- (2) do-연산자: “공부 시간을 **강제로** 5시간으로 만들었을 때의 평균 점수”

<br>

# 4. Backdoor Criterion (역인과 경로 차단)

**혼란(confounding)**을 조절하기 위한 핵심 도구!!!!

<br>

변수 집합 $$Z$$가 $$X \rightarrow Y$$ 경로의 **backdoor path**를 모두 차단하면,

$$\rightarrow$$ $$P(Y \mid \text{do}(X)) = \sum_z P(Y \mid X, Z=z)P(Z=z)$$

즉, **관측된 데이터만으로** 인과 효과 추정 가능!!

<br>

### Example 1

```
graph LR
U[지능 U] --> X[공부시간 X]
U --> Y[시험 점수 Y]
X --> Y
```

- 이 경우 U는 **confounder** (원인과 결과 모두에 영향을 주는 변수)

- $$P(Y \mid \text{do}(X)) \neq P(Y \mid X)$$.

  → $$U$$를 통제해야 인과 추론 가능

<br>

### Example 2

- $$X$$: 운동 여부 (0 = 안 함, 1 = 함)
- $$Y$$: 건강 점수 (0~100)
- $$Z$$: 나이 (confounder, 0 = 젊음, 1 = 노년)
- DAG:

```
Z → X → Y
Z → Y
```

| **Z(나이)** | **X(운동)** | **Y(건강 점수)** |
| ----------- | ----------- | ---------------- |
| 0           | 1           | 90               |
| 0           | 0           | 85               |
| 1           | 1           | 70               |
| 1           | 0           | 60               |

1. 나이별 조건부 기대값:

   - $$P(Y \mid X=1, Z=0) = 90, P(Y \mid X=0, Z=0) = 85$$.
   - $$P(Y \mid X=1, Z=1) = 70, P(Y \mid X=0, Z=1) = 60$$>

2. 나이 분포: $$P(Z=0)=0.5, P(Z=1)=0.5$$

3. **Backdoor Adjustment**

   - $$P(Y \mid \text{do}(X=1)) = 0.5 \times 90 + 0.5 \times 70 = 80$$.
   - $$P(Y \mid \text{do}(X=0)) = 0.5 \times 85 + 0.5 \times 60 = 72.5$$.

   $$\rightarrow$$ $$\text{ATE} = 80 - 72.5 = 7.5$$.

<br>

# 5. RCM vs. SCM

| **항목**                | **Rubin Causal Model (RCM)**   | **Structural Causal Model (SCM)** |
| ----------------------- | ------------------------------ | --------------------------------- |
| 기반                    | 잠재 결과 (Potential Outcomes) | 구조적 함수와 그래프              |
| 인과 효과 정의          | $$Y(1) - Y(0)$$                | $$P(Y \mid \text{do}(X))$$        |
| 인과 추정 방법          | 무작위화, 매칭, 회귀 등        | 도구 변수, 백도어 조정 등         |
| 표현 방식               | 수식 기반, 잠재 결과           | 인과 그래프 (DAG), 방정식         |
| 반사실(Counterfactuals) | 있음                           | 매우 정교하게 다룸                |

