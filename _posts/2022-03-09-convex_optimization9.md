---
title: (09) Lagrangian
categories: [CO]
tags: [Convex Optimization]
excerpt: (참고) 모두를 위한 convex optimization

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# 09. Lagrangian

( 반드시 “convex” problem일 필요는 없음 )

<br>

## 9-1. Introduction

**(1) Problem** : **제약 조건** 이 있는 최적화 문제

$$\begin{array}{cl}
\min _{x} & f(x) \\
s . t . & h_{i}(x) \leq 0, i=1, \ldots, m \\
& l_{j}(x)=0, j=1, \ldots, r
\end{array}$$.

<br>

**(2) Lagrangian**

$$L(x, u, v)=f(x)+\sum_{i=1}^{m} u_{i} h_{i}(x)+\sum_{j=1}^{r} v_{j} l_{j}(x)$$.

- Lagrangian 승수
  - $$u$$ : inequality constraint $$h$$에 해당하는 Lagrangian 승수
    - ($$h$$ 의 개수만큼 존재) $$u \in \mathbb{R}^{m}$$
  - $$v$$ : equality constraint $$l$$에 해당하는 Lagrangian 승수
    - ($$l$$ 의 개수만큼 존재) $$ v \in \mathbb{R}^{r}$$

<br>

**(3) Objective Function & Lagrangian의 관계**

$$L(x, u, v)=f(x)+\sum_{i=1}^{m} u_{i} \underbrace{h_{i}(x)}_{\leq 0}+\sum_{j=1}^{r} v_{j} \underbrace{l_{j}(x)}_{=0} \leq f(x)$$.

- 즉, Lagrangian은 Objective Function의 **하한 (lower bound)** 이다.

<br>

## 9-2. Lagrangian Dual Function

Notation

- $$C$$ : primal feasible set
- $$f^{*}$$ : primal 최적값

<br>

Lower Bound

- $$f^{*} \geq \min _{x \in C} L(x, u, v) \geq \min _{x} L(x, u, v):=g(u, v)$$.
  - 여기서 $$g(u,v)$$는 **Lagrange Dual Function** 이다.
  - $$\lambda$$ ( 라그랑즈 승수 ) = $$(u,v)$$

<br>

### Ex) Quadratic Program

**(1) Problem**

$$\min _{x} \frac{1}{2} x^{T} Q x+c^{T} x$$,

such that

- $$Ax=b$$,

- $$x \geq 0$$.

<br>

**(2) Lagrangian**

$$L(x, u, v)=\frac{1}{2} x^{T} Q x+c^{T} x-u^{T} x+v^{T}(A x-b)$$.

- $$x \geq 0$$ 이므로, $$u^T$$ 가 아닌 $$-u^T$$ 를 곱해준다.

  ( 하한을 만들 것이므로, $$+$$ & $$-$$의 곱이 되게끔 )

$$\rightarrow$$ 이 Lagrangian Function을 minimize하는 방향으로 문제를 풀 것이다.

<br>

**(3) Lagrangian Dual Function**

- Lagrangian Function을 minimize하기 위해, $$L$$을 $$x$$ 에 대해 미분하고 0으로 만드는 $$x^{*}$$ 를 구할 것이다.
  - $$Q x-\left(c-u+A^{T} v\right)=0$$.
  - $$x^{*} = Q^{-1}\left(c-u+A^{T} v\right)$$,

$$\rightarrow$$ 이 $$x^{*}$$ 를 Lagrangian $$L(x,u,v)$$에 대입을 하면, 아래와 같다.

$$-\frac{1}{2}\left(c-u+A^{T} v\right)^{T} Q^{-1}\left(c-u+A^{T} v\right)-b^{T} v$$.

<br>

위 식이 곧 **Lagrangian Dual Function / $$g(u,v)$$  / 하한** 이다

- $$g(u, v)=\min _{x} L(x, u, v)=-\frac{1}{2}\left(c-u+A^{T} v\right)^{T} Q^{-1}\left(c-u+A^{T} v\right)-b^{T} v$$.
- 이 하한을 **최대화** 함으로써, 가장 좋은 하한을 얻으낼 수 있다.

<br>

## 9-3. Lagrange Dual Problem

**(1) Problem**

$$\begin{array}{cl}
\min _{x} & f(x) \\
s . t . & h_{i}(x) \leq 0, i=1, \ldots, m \\
& l_{j}(x)=0, j=1, \ldots, r
\end{array}$$.

<br>

**(2) Dual Function $$g(u,v)$$**

- 특징 : 모든 $$u \geq 0$$ & $$v$$에 대해, $$f^{*} \geq g(u,v)$$를 만족

- 위 조건을 만족하는 (=feasible) $$u,v$$에 대해 $$g(u,v)$$를 최대화 하기!

  $$\rightarrow$$ **Lagrangian Dual Problem**

<br>

**(3) Lagrangian Dual Problem**

$$\begin{aligned}
&\max _{u, v} \quad g(u, v) \\
&\text { s.t. } \quad u \geq 0
\end{aligned}$$.

- dual 최적값 : $$g^{*}$$
- 이때, $$f^{*} \geq g^{*}$$를 만족한다 ( = weak duality )
- 특징 정리
  - (1) weak duality는 convex problem이 아니어도 항상 성립한다
  - (2) primal problem이 convex problem이 아니어도, dual problem은 convex problem 이다

$$\begin{aligned}
g(u, v) &=\min _{x}\left\{f(x)+\sum_{i=1}^{m} u_{i} h_{i}(x)+\sum_{j=1}^{r} v_{j} l_{j}(x)\right\} \\
&=\underbrace{-\max _{x}\left\{-f(x)-\sum_{i=1}^{m} u_{i} h_{i}(x)-\sum_{j=1}^{r} v_{j} l_{j}(x)\right\}}_{\text {pointwise maximum of convex functions in }(u, v)}
\end{aligned}$$.

<br>

## 9-4. Strong Duality

Weak & Strong duality

- weak duality : $$f^{*} \geq g^{*}$$
- strong duality : $$f^{*} = g^{*}$$

<br>

(1) Slater Condition

( Strong Duality를 만족하기 위한 충분조건 )

- Primal problem이 convex problem이고,

  Strictly feasible한 $$x \in R^{n}$$이 한개 이상 있으면, Strong duality 만족!

  ( $$h_{1}(x)<0, \ldots, h_{m}(x)<0, \text { and } l_{1}(x)=0, \ldots, l_{r}(x)=0$$ )

<br>

### Ex) SVM Dual problem

Notation

- $$y \in\{-1,1\}^{n}, X \in \mathbb{R}^{n \times p}$$.

<br>

(1) SVM problem

$$\min _{\beta, \beta_{0}, \xi} \frac{1}{2} \mid \mid \beta \mid \mid _{2}^{2}+C \sum_{i=1}^{n} \xi_{i}$$.

such that

- $$\xi_{i} \geq 0, i=1, \ldots, n$$,

- $$y_{i}\left(x_{i}^{T} \beta+\beta_{o}\right) \geq 1-\xi_{i}, i=1, \ldots, n$$.

  ( $$1-\xi_{i}-y_{i}(x_{i}^{T} \beta+\beta_{o}) \leq 0$$ 으로 바꿔 쓸 수 있음 )

<br>

(2) Lagrangian

$$L\left(\beta, \beta_{0}, \xi, v, w\right)=\frac{1}{2} \mid \mid \beta \mid \mid _{2}^{2}+C \sum_{i=1}^{n} \xi_{i}-\sum_{i=1}^{n} v_{i} \xi_{i}+\sum_{i=1}^{n} w_{i}\left(1-\xi_{i}-y_{i}\left(x_{i}^{T} \beta+\beta_{o}\right)\right)$$.

위 $$L$$을 최소화하는 

- (a) $$\beta$$
- (b) $$\beta_0$$

- (c) $$\xi$$ 

를 각각 구한 뒤, $$L$$에 다시 대입한다. 그것이 곧 Lagrangian Dual Function 이다.

<br>

(3) Lagrangian Dual Function

$$g(v, w)= \begin{cases}-\frac{1}{2} w^{T} \tilde{X} \tilde{X}^{T} w+1^{T} w, & \text { if } w=C 1-v, w^{T} y=0 \\ -\infty, & \text { otherwise }\end{cases}$$.

- where $$\tilde{X}=\operatorname{diag}(y) X$$

<br>

(4) Dual Problem

- 위의 Lagrangian Dual function을 maximize하는 문제
- ( slack variable인 $$v$$ 제거 )

$$\begin{array}{ll}
\max _{w} & -\frac{1}{2} w^{T} \tilde{X} \tilde{X}^{T} w+1^{T} w \\
\text { s.t. } & 0 \leq w \leq C 1, w^{T} y=0
\end{array}$$.

<br>

(5) Strong Duality

- Primal Problem이 Slater 조건을 만족한다. 그 말은 즉,

  - (1) Primal Problem의 objective function이 **convex** 이고
  - (2) Inequality constraint가 $$\beta, \beta_0, \xi$$ 에 대한 affine transformation 이다.

- 위 두 조건을 만족하기 때문에 ( = Slater 조건을 만족하기 때문에 ),

  SVM dual problem은 **Strong Duality**가 성립된다.

<br>

## 9-5. Duality Gap

Duality Gap : $$f(x) - g(u,v)$$

앞서 공부했듯, 아래의 조건은 항상 만족한다.

- $$f(x) - f^{*} \leq f(x) - g(u,v)$$.

<br>

따라서, duality gap이 0이 되면, 

- $$x$$는 Primal problem의 최적해
- $$u,v$$는 Dual problem의 최적해

