---
title: Causal Inference - Part 2
categories: [ML, CI]
tags: []
excerpt: Causal Graph, DAG, 조건부 독립성, 충돌 변수
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Causal Inference - Part 2

## Contents

1. 인과 다이어그램 (Causal Graph / DAG)
2. 조건부 독립성
3. 충돌 변수
4. 인과 추론의 목표
5. 인과 추론에서의 편향
6. 인과 추론의 다양한 프레임워크
7. Rubin Causal Model
8. Structural Causal Model

<br>

# 1. 인과 다이어그램 (Causal Graph / DAG)

## (1) Causal Graph란

- 인과 구조를 시각적으로 표현하는 방법
- **DAG (Directed Acyclic Graph)**를 사용
  - 이를 통해, 어떤 변수들을 통제해야(confounder control) 하는지 알 수 있음!

- 예시

  ```
  나이  →  건강 상태  → 회복 여부
         ↘ 신약 복용 ↗
  ```

<br>

## (2) 핵심 요소

- **Confounder (혼란변수)**: 원인과 결과에 모두 영향을 주는 변수 → 반드시 통제 필요
- **Mediator (매개변수)**: 원인과 결과 사이의 경로에 있는 변수 → 경우에 따라 통제하면 안 됨
- **Collider (충돌변수)**: 두 원인이 만나는 지점 → 통제하면 편향 생김

<br>

# 2. 조건부 독립성 (Conditional Independence)

- **인과 그래프(DAG)**를 분석하거나, 어떤 변수를 통제해야 하는지 결정할 때 필수!

- 정의: 어떤 두 변수 X, Y가 **세 번째 변수 Z를 알고 나면 서로 독립**이 되는 것

  ```
  X ⫫ Y | Z
  ```

- 예시:

  ```
  우산 ⫫ 고인 물 | 비 여부  
  (X ⫫ Y | Z)
  ```

- 시각화 with DAG

  ```
  X ← Z → Y
  ```

  - Z: 공통 원인(confounder)

  - X, Y는 Z 때문에 상관관계가 있음

  - 하지만 **Z를 통제하면** X와 Y는 독립 → 이걸 **Backdoor Path 차단**이라고도 함!

    ( 인과 추론에서 이걸 찾아서 통제하는 게 핵심 )

<br>

# 3. Collider (충돌 변수)

- 정의: Z는 두 원인의 결과 (ex: X: 운동, Y: 식단 → Z: 체중)

  ```
  X → Z ← Y
  ```

- Z를 **통제하면 오히려 X와 Y 사이에 인위적인 상관관계**가 생김

→ 이건 **조건부 의존성(induced dependence)**!!

→ 즉, **Z를 통제하면 안 되는 경우도 있다!**

<br>

## Summary

| **개념**             | **설명**                                             |
| -------------------- | ---------------------------------------------------- |
| 조건부 독립성        | Z를 알고 나면 X와 Y는 독립이다: X ⫫ Y                |
| 인과 추론에서의 역할 | 어떤 변수를 통제해야 confounding이 사라지는지 알려줌 |
| Backdoor 차단        | 조건부 독립성 확보를 통해 인과 효과를 추정 가능      |
| Collider 경로        | 통제하면 안 되는 경우 (조건부 의존성 유발)           |

<br>

# 4. 인과 추론의 목표

- **ATE (Average Treatment Effect)**: 전체 집단에서 평균적인 인과 효과

- **ATT (Average Treatment effect on the Treated)**: 처치를 받은 집단에 대해 인과 효과

- **CATE (Conditional ATE)**: 특정 조건 하의 인과 효과 (예: 고령자에 대한 효과만)

<br>

중요한 이유

- 정책 결정에서 “모두에게 효과가 있는가?” vs. “특정 그룹에만 효과가 있는가?“는 다름
- 예: 신약이 노인에게만 효과적이라면, ATE는 낮아도 CATE는 높을 수 있음

<br>

# 5. 인과 추론에서의 편향 (Bias)

편향의 종류

| **편향 유형**      | **설명**                        |
| ------------------ | ------------------------------- |
| **선택 편향**      | 처치 여부가 무작위가 아닐 때    |
| **측정 오류**      | 변수 값이 잘못 기록되었을 때    |
| **혼란 편향**      | confounder를 통제하지 않았을 때 |
| **모형 지정 편향** | 잘못된 회귀 모형을 썼을 때      |

<br>

# 6. 인과 추론의 다양한 프레임워크

## (1) Rubin Causal Model (Potential Outcomes Framework)

(2) Structural Causal Model (SCM, by Judea Pearl)

- 개념: 앞선 ATE, ATT 등의 개념을 수학적으로 표현
- 전제: ***“각 개인에게 두 개의 잠재적 결과가 존재한다”***

<br>

## (2) Structural Causal Model (SCM, by Judea Pearl)

- DAG, do-calculus 등 구조적 접근 강조
- 인과 다이어그램을 통해 수학적 추론이 가능

<br>

# 7. Rubin Causal Model 

***“한 사람에게 서로 다른 처치를 했을 때 결과가 어떻게 달랐을까?“***를 중심으로 인과를 정의

<br>

## (1) Potential Outcomes (잠재 결과)

각 개인은 두 개의 결과를 가짐:

- $$Y(1)$$: 처치를 받았을 때의 결과
- $$Y(0)$$: 처치를 받지 않았을 때의 결과

$$\rightarrow$$ 단, 현실에서는 **둘 중 하나만 관측** 가능!

<br>

## (2) Causal Effect = $$Y(1) - Y(0)$$

- 정의: 두 잠재 결과의 차이로 정의됨

- 개인 수준에서 직접 관측 불가능 ( 두 세계관 불가능! )

  → 평균적인 효과(ATE, ATT 등)를 추정!

<br>

## (3) 처치 할당과 독립성 가정 (Ignorability / Unconfoundedness)

한 줄 요약: ***무작위 실험을 하지 않았는데도, 관측된 변수 X를 잘 통제하면 마치 실험처럼 인과 추정이 가능하다***

```
(Y(0), Y(1)) ⫫ T | X
```

- 즉, 관측된 변수 X를 통제하면 처치가 무작위처럼 된다



## (4) 무작위 실험이 이론적 이상

RCT는 잠재 결과와 처치 간 독립성을 자동으로 보장

$$\therefore$$ 그래서 Rubin 모델은 실험 기반 설계와 잘 어울림

<br>

## (5) 통계적 추론 중심

- 인과 추론을 **통계적 추정 문제**로 접근
- 회귀, 매칭, 성향 점수 등으로 잠재 결과의 차이를 추정

<br>

# 8. Structural Causal Model (SCM)

***“세상은 변수들 사이의 인과적 구조로 작동한다”***는 철학

→ 인과는 함수 관계와 그래프로 정의!

<br>

## (1) 인과 그래프 (DAG: Directed Acyclic Graph)

- 변수 간 인과 관계 = **화살표**
  - 모든 인과 추론은 이 구조 위에서 해석

<br>

## (2) 구조 방정식 (Structural Equations)

- 각 변수는 **"다른 변수의 함수"**로 표현
  - ex) $$Y = f(X, U)$$
  - 함수와 외생 변수(U)를 통해 **결정론적 인과 모델** 정의

<br>

## (3) do-연산과 인과 효과 정의

- 개입을 수학적으로 표현:$$ P(Y \mid do(X))$$

- do는 “X를 강제로 x로 만든다”는 의미 

  → 단순 조건부 확률과 다름!!

<br>

## (4) 식별성과 도출 규칙 (do-calculus)

- 관측된 데이터만으로 인과 효과를 추정할 수 있는 조건을 도출
- **do-calculus**를 통해 P(Y | do(X))를 관측 가능한 확률식으로 바꾸는 기법

<br>

## (5) 관측, 개입, 반사실 추론을 모두 포함

세 가지 질문에 모두 답하려는 포괄적 인과 이론

1. 관찰: $$P(Y \mid X)$$
2. 개입: $$P(Y \mid do(X))$$
3. 반사실: $$P(Y_x \mid X = x’, Y = y’)$$

<br>

## Summary

| **요소**  | **Rubin Model**     | **Structural Causal Model** |
| --------- | ------------------- | --------------------------- |
| 인과 정의 | 잠재 결과 간 차이   | 구조 방정식과 개입 (do)     |
| 주 도구   | 통계 추정, RCT 기반 | DAG, do-calculus            |
| 변수 관계 | 확률적              | 함수적 (결정론 + 확률)      |
| 관점      | 통계적 사고         | 인과적 구조 모델링          |
| 강점      | 실험 설계 및 추정   | 개입, 반사실까지 해석 가능  |

