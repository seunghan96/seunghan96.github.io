---
title: Causal Inference - Part 3
categories: [ML, TS]
tags: []
excerpt: Rubin Causal Model (RCM) 상세
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Causal Inference - Part 3

## Contents

1. Rubin Causal Model (RCM)
2. 무작위 실험 (Randomized Experiment)
3. 통계적 방법: 매칭 (Matching)
4. 통계적 방법: 회귀(Regression Adjustment)
5. Comparison

<br>

# 1. Rubin Causal Model (RCM)

- **Potential Outcomes Framework**라고도 부름
- 핵심: **“무엇이 일어났을까?”와 “무엇이 일어났을 수도 있었을까?”**를 비교함으로써 인과 효과를 정의합



## (1) 핵심 개념

### a) **Potential Outcomes (잠재 결과)**

- 어떤 단위 (unit, 예: 사람, 회사 등)에 대해 두 개의 잠재 결과가 존재

  - Y(1): 처치를 받았을 때의 결과 (treated)
  - Y(0): 처치를 받지 않았을 때의 결과 (control)

- 단위당 ***하나의 결과만*** 실제로 관측됨 

  $$\rightarrow$$   **“Fundamental Problem of Causal Inference”**:

<br>

### b) **Causal Effect (인과 효과)**

- 한 단위에서의 인과 효과:

  $$\text{Causal Effect} = Y(1) - Y(0)$$

- 하지만 위에서 본 것처럼 두 결과를 모두 볼 수 없음

  $$\rightarrow$$ 따라서, **집단 평균 효과(Average Treatment Effect, ATE)**를 추정!

<br>

### c) **ATE (Average Treatment Effect)**

$$\text{ATE} = \mathbb{E}[Y(1) - Y(0)] = \mathbb{E}[Y(1)] - \mathbb{E}[Y(0)]$$.

<br>

## (2) **Assumptions (전제)**

1. **Stable Unit Treatment Value Assumption (SUTVA)**:
   - 다른 단위의 처리가 내 결과에 영향을 주지 않음 (no interference)
   - 처치가 한 형태로만 정의됨 (well-defined treatment)
2. **Ignorability (또는 Unconfoundedness)**:
   - $$(Y(1), Y(0)) \perp T \mid X$$
     - 즉, 조건부로 무작위 할당이라면 인과 효과 추정 가능
3. **Overlap (Common Support)**:
   - 어떤 X 값에서도 처치군과 대조군이 모두 존재해야 함.

<br>

## (3) 예시

| **학생** | **보충수업 (T)** | **실제 성적 (Y)** | **예상 성적 if no 수업 (Y(0))** | **예상 성적 if 수업 (Y(1))** |
| -------- | ---------------- | ----------------- | ------------------------------- | ---------------------------- |
| A        | 1                | 85                | 80                              | 85                           |
| B        | 1                | 90                | 85                              | 90                           |
| C        | 1                | 75                | 70                              | 75                           |
| D        | 0                | 78                | 78                              | 82                           |
| E        | 0                | 82                | 82                              | 86                           |
| F        | 0                | 70                | 70                              | 75                           |



### **잠재 결과와 인과 효과**

- A의 인과 효과: $$85 - 80 = 5$$
- ...
- D의 인과 효과: $$82 - 78 = 4$$
- ..

$$\therefore$$ 전체 ATE:

$$\frac{(5 + 5 + 5 + 4 + 4 + 5)}{6} = \frac{28}{6} \approx 4.67$$.

<br>

### Summary

Rubin Causal Model은 잠재 결과 중 ***하나만 관측 가능***하므로...

$$\rightarrow$$ 인과 효과를 정확히 추정하기 위해서는 **무작위 실험**이나 **통계적 방법**을 사용해야!!

<br>

# 2. 무작위 실험 (Randomized Experiment)

| **학생** | **보충수업 (T)** | **실제 성적 (Y)** |
| -------- | ---------------- | ----------------- |
| A        | 1                | 85                |
| B        | 1                | 90                |
| C        | 1                | 75                |
| D        | 0                | 78                |
| E        | 0                | 82                |
| F        | 0                | 70                |

- **처치군 평균**: $$\frac{85 + 90 + 75}{3} = \frac{250}{3} \approx 83.3$$
- **대조군 평균**: $$\frac{78 + 82 + 70}{3} = \frac{230}{3} \approx 76.7$$
- **ATE 추정**: $$83.3 - 76.7 = 6.6$$

<br>

무작위 할당 덕분에

- $$\mathbb{E}[Y(1) \mid T=1] \approx \mathbb{E}[Y(1)]$$,

- $$\mathbb{E}[Y(0) \mid T=0] \approx \mathbb{E}[Y(0)]$$,

$$\rightarrow \therefore $$ 단순 평균 차이로도 ATE를 추정 가능!

<br>

# 3. 통계적 방법: 매칭 (Matching)

비슷한 사람끼리 비교하자!

e.g., **보충수업을 받은 학생**과 **비슷한 성향의 안 받은 학생**을 짝지어!

| **보충수업 학생** | **성적 (Y)** | **매칭된 학생** | **성적 (Y)** | **추정 인과 효과** |
| ----------------- | ------------ | --------------- | ------------ | ------------------ |
| A                 | 85           | D (78점)        | 78           | 7                  |
| B                 | 90           | E (82점)        | 82           | 8                  |
| C                 | 75           | F (70점)        | 70           | 5                  |

- 추정 ATE: $$\frac{(7 + 8 + 5)}{3} = \frac{20}{3} \approx 6.67$$

<br>

# 4. **통계적 방법: 회귀(Regression Adjustment)**

- 보충수업 여부 $$T$$ & 다른 공변량 $$X$$ (e.g., 사전 성적 등)

- $$Y = \alpha + \beta T + \gamma X + \epsilon$$.
  - $$\beta$$: 보충수업의 **추정된 인과 효과**

<br>

### Example

| **학생** | **T (보충수업)** | **X (사전 점수)** | **Y (최종 점수)** |
| -------- | ---------------- | ----------------- | ----------------- |
| A        | 1                | 80                | 85                |
| B        | 1                | 85                | 90                |
| C        | 1                | 70                | 75                |
| D        | 0                | 78                | 78                |
| E        | 0                | 82                | 82                |
| F        | 0                | 70                | 70                |

- $$\beta \approx 5$$ 로 추정됨

<br>

한 줄 요약: ***공변량 X를 통제***함으로써 selection bias를 완화하고 인과 효과를 추정하는 방식입니다.

<br>

# 5. Comparison

| **방법**    | **장점**                    | **한계**                    |
| ----------- | --------------------------- | --------------------------- |
| 무작위 실험 | 인과 추론의 황금 기준       | 현실적으로 어려운 경우 많음 |
| 매칭        | 직관적이고 해석 쉬움        | 매칭 잘 안 될 수도 있음     |
| 회귀        | 일반적, 많은 상황 적용 가능 | 모델 가정에 민감함          |

