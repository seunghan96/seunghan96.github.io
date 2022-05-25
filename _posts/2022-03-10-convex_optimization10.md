---
title: (10) KKT condition
categories: [CO]
tags: [Convex Optimization]
excerpt: (참고) 모두를 위한 convex optimization
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# 10. KKT condition

## 10-1. Introduction

( Primal problem이 convex일 때 )

- KKT condition : primal & dual optimal points에 대한 충분조건

<br>

또한,

- (1) Primal Problem의 Objective Function & Constraint가 미분가능하고

- (2) Strong Duality를 만족할 때

$$\rightarrow$$ 항상 KKT condition 만족

<br>

KKT condition 덕에, 많은 문제들이 analytically 풀림

<br>

## 10-2. KKT condition이란

( 일반적인 Primal Problem )

$$\min _{x} f(x)$$,

subject to 

- $$h_{i}(x) \leq 0, i=1, \ldots, m$$,
- $$l_{j}(x)=0, j=1, \ldots, r$$.

<br>

### KKT Condition

- (1) Stationarity
- (2) Complementary Slackness
- (3) Primal Feasibility
- (4) Dual Feasibility

<br>

**(1) Stationarity**

- $$0 \in \partial\left(f(x)+\sum_{i=1}^{m} \lambda_{i} h_{i}(x)+\sum_{j=1}^{r} \nu_{j} l_{j}(x)\right)$$.
- 의미 : $$\lambda, \nu$$ 고정 시, $$x$$ 에 대한 미분이 $$0$$ 을 포함

<br>

**(2) Complementary Slackness**

- $$\lambda_{i} \cdot h_{i}(x)=0$$.
- 의미 : $$\lambda_{i}$$ 와 $$h_{i}$$ 중 적어도 하나는 0 

<br>

**(3) Primal Feasibility**

- $$h_{i}(x) \leq 0, l_{j}(x)=0$$ for all $$i, j$$ 
- Primal problem의 제약조건들을 만족함

<br>

**(4) Dual Feasibility**

- $$\lambda_{i} \geq 0$$ for all $$i$$
- Dual problem의 제약조건들을 만족함

<br>

### Sufficiency

( Primal Problem = convex problem 하에 … )

KKT condition을 만족하는 $$x^{*}, \lambda^{*}, \nu^{*}$$가 있을 때

$$\rightarrow$$ $$x^{*}, \lambda^{*}, \nu^{*}$$ 는 **Zero duality gap의 primal & dual solution이다!**

<br>

### Neccessity

$$x^{*}, \lambda^{*}, \nu^{*}$$ 가 Zero duality gap ( + Slater Condition 만족  = Strong Duality ) 의 primal & dual solution일 때,

$$\rightarrow$$ **KKT condition을 만족**한다

<br>

### [ SUMMARY ]

KKT condition

- Zero duality gap의 primal & dual solution에 대한 충분조건
- Strong Duality를 만족한다면, primal & dual solution에 대한 필요조건

<br>

따라서, strong duality 를 만족하면, 이 둘은 “필요충분조건”

$$x^{\star}, \lambda^{\star}, \nu^{\star}$$ are primal and dual solutions
$$\Leftrightarrow x^{\star}, \lambda^{\star}, \nu^{\star}$$ satisfy the KKT conditions

<br>

## 10-3. Ex) Quadratic with Equality Constraints

Notation :  $$P \in \mathbb{S}_{+}^{n}$$ and $$A \in \mathbb{R}^{\mathrm{pxn}}$$

<br>

(1) Problem

$$\min _{x}(1 / 2) x^{T} P x+q^{T} x+r$$.

subject to

- $$A x=b$$.

<br>

(2) KKT Condition

- **a) Stationarity**
  - $$P x^{\star}+q+A^{T} \nu^{\star}=0$$.
- **b) Complementary Slackness**
  - inequality constraint 없으므로 고려 대상 X
- **c) Primal feasibility**
  - $$A x^{\star}=b$$.
- **d) Dual feasibility**
  - inequality constraint 없으므로, 제약 걸린 lagrange multiple 없으므로 고려 대상 X

<br>

(3) KKT matrix

- 위의 문제를, block matrix 형태로 나타낸 것

$$\left[\begin{array}{ll}
P & A^{T} \\
A & 0
\end{array}\right]\left[\begin{array}{c}
x^{\star} \\
\nu^{\star}
\end{array}\right]=\left[\begin{array}{c}
-q \\
b
\end{array}\right]$$.

<br>

위 식을 풀면, primal & dual solution을 구할 수 있다.

<br>

## 10-4. Ex) Water Filling problem

(1) Problem

$$\min _{x}-\sum_{i=1}^{n} \log \left(\alpha_{i}+x_{i}\right)$$,

subject to 

- $$x \succeq 0,1^{T} x=1$$,
- $$\alpha_{i}>0$$.

<br>

의미

- (1) $$n$$ 개의 채널에 전력 할당

- (2) $$x_i$$ : $$i$$ 번째 채널에 할당되는 송신기의 출력

- (3) $$\log (\alpha_i + x_i)$$ : $$i$$번째 채널의 capacity ( communication rate )

  $$\rightarrow$$ **모든 채널에 대해, 이 capacity를 최대화하기위해, 각 채널에 얼만큼 할당해야할지의 task**

<br>

Lagrange multiplier

- $$\lambda^{\star} \in \mathbb{R}^{n}$$ : inequality constraint $$x^{\star} \succeq 0$$ 에 대한 ~
- $$\nu^{\star} \in \mathbb{R}$$ : equality constraint $$1^{T} x^{\star}=1$$ 에 대한 ~

<br>

(2) KKT Condition

**a) Stationarity**

- ( $$\lambda, \nu$$ 고정 시, $$x$$ 에 대한 미분이 $$0$$ 을 포함 )
- $$-1 /\left(\alpha_{i}+x_{i}^{\star}\right)-\lambda_{i}^{\star}+\nu^{\star}=0, i =1, \ldots, n$$.

<br>

**b) Complementary Slackness**

- ( $$\lambda_{i}$$ 와 $$h_{i}$$ 중 적어도 하나는 0 )
- $$\lambda_{i}^{\star} x_{i}^{\star} =0, i=1, \ldots, n$$.

<br>

**c) Primal Feasibility**

- ( Primal problem의 제약조건들을 만족함 )
- $$1^{T} x^{\star}=1$$.
- $$x^{\star} \succeq 0$$.

<br>

**d) Dual Feasibility**

- ( Dual problem의 제약조건들을 만족함 )
- $$\lambda^{\star} \succeq 0$$.

<br>

**a) Stationarity** + **b) Complementary Slackness** 에 의해,

$$\rightarrow$$ $$x_{i}^{\star}=\left\{\begin{array}{ll}
1 / \nu^{\star}-\alpha_{i} & \nu^{\star}<1 / \alpha_{i} \\
0 & \nu^{\star} \geq 1 / \alpha_{i}
\end{array}=\max \left\{0,1 / \nu^{\star}-\alpha_{i}\right\}, \quad i=1, \ldots, n .\right.$$.

<br>

**c) Primal Feasibility**에 의해,

$$\rightarrow$$ $$\sum_{i=1}^{n} \max \left\{0,1 / \nu^{\star}-\alpha_{i}\right\}=1$$

- $$1/\nu^{*}$$ 에 대한 piecewise-linear increasing function

  따라서, 고정된 $$\alpha_i$$에 대한 unique solution을 가짐

<br>

## 10-5. ex) SVM

Notation : $$y \in\{-1,1\}^{n}$$ and $$X \in \mathbb{R}^{n \times p}$$.

<br>

(1) Goal :

$$\min _{\beta, \beta_0, \xi} \quad \frac{1}{2} \mid \mid \beta \mid \mid _{2}^{2}+C \sum_{i=1}^{n} \xi_{i}$$.

subject to 

- $$\xi_{i} \geq 0, \quad i=1, \ldots, n$$
- $$y_{i}\left(x_{i}^{T} \beta+\beta-0\right) \geq 1-\xi_{i}, \quad i=1, \ldots, n,$$.

<br>

(2) Lagrangian function

- Lagrangian multiplier : $$v^{*}$$ ( 1번째 inequality ), $$w^{*}$$ ( 2번째 inequality )

$$L\left(\beta, \beta-0, \xi, v^{\star}, w^{\star}\right)=\frac{1}{2} \mid \mid \beta \mid \mid _{2}^{2}+C \sum_{i=1}^{n} \xi_{i}-\sum_{i=1}^{n} v_{i}^{\star} \xi_{i}+\sum_{i=1}^{n} w_{i}^{\star}\left(1-\xi_{i}-y_{i}\left(x_{i}^{T} \beta+\beta_{0}\right)\right.$$.

<br>

(3) KKT Condition

- **a) Stationarity** : $$\beta, \beta_0, \xi$$ 에 대해 미분값=0 되도록
  - $$0=\sum_{i=1}^{n} w_{i}^{\star} y_{i}$$.
  - $$\beta=\sum_{i=1}^{n} w_{i}^{\star} y_{i} x_{i}$$,
  - $$w^{\star}=C \cdot 1-v^{\star}$$.
- **b) Complementary Slackness** : ( $$\lambda_{i}$$ 와 $$h_{i}$$ 중 적어도 하나는 0 )
  - $$v_{i}^{\star} \xi_{i}=0$$,
  - $$w_{i}^{\star}\left(1-\xi_{i}-y_{i}\left(x_{i}^{T} \beta+\beta-0\right)\right)=0, \quad 1=1, \ldots, n$$.

<br>

결론 :

- optimal ) $$\beta^{\star}=\sum_{i=1}^{n} w_{i}^{\star} y_{i} x_{i}$$
- support point란?
  - $$y_{i}\left(x_{i}^{T} \beta^{\star}+\beta-0^{\star}\right)=1-\xi_{i}^{\star}$$ 를 만족하는 점
- 어떤 support point $$i$$에 대해,
  - $$\xi_{i}^{\star} = 0$$ $$\rightarrow$$ $$x_i$$ 는 “hyperplane 상”에 위치 
    - (1) $$\xi_{i}^{\star} = 0$$ 이므로, $$v^{\star}$$ 는 0이 아닐 수도 있다……… ( $$v_{i}^{\star} \xi_{i}=0$$ 이므로 ) 
    - (2)  $$w_{i}^{\star} \in(0, C]$$ ………. ( $$w^{\star}=C \cdot 1-v^{\star}$$  이므로 )
  - $$\xi_{i}^{\star} \neq 0$$ $$\rightarrow$$ $$x_i$$는 “hyperplane 반대쪽” ( = 어김 )에 위치
    - (1) $$\xi_{i}^{\star} \neq 0$$ 이므로, $$v^{\star}$$ 는 0일 수 밖에 없다 ……… ( $$v_{i}^{\star} \xi_{i}=0$$ 이므로 ) 
    - (2)  $$w_{i}^{\star} = C$$ ………. ( $$w^{\star}=C \cdot 1-v^{\star}$$  이므로 )

<br>

![figure2](/assets/img/co/img32.png)

