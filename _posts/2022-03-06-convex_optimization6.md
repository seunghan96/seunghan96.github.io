---
title: (06) Gradient Descent
categories: [CO]
tags: [Convex Optimization]
excerpt: (참고) 모두를 위한 convex optimization
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# 06. Gradient Descent

![figure2](/assets/img/co/img22.png)

<br>

## 6-1. Gradient Descent 란

$\min _{x} f(x), f:$ differentiable with $\operatorname{dom}(f)=R^{n}$

특징

- convex & differentiable $f$

- 제약조건 X

<br>

Notation

- optimal value : $f^{*}=\min _{x} f(x)$
- optimal : $x^{*}$ 

<br>

### Process ( 표현 1 )

Step 1. 초기 점 $x^{(0)} \in R^{n}$ 을 선택

Step 2. 아래를 반복

- $x^{(k)}=x^{(k-1)}-t_{k} \nabla f\left(x^{(k-1)}\right), k=1,2,3, \ldots, t_{k}>0$.
  - $t_k$ : learning rate / step size ..

<br>

### Process ( 표현 2 )

Step 1. 초기점 $\mathrm{x}, x \in \operatorname{dom} f$ 선택

Step 2. 아래를 반복

- (2-1) descent direction $\Delta x=-\nabla f(x)$ 찾기
- (2-2) Line Search
  - step size $t>0$ 정하기
- (2-3) $x=x+t \Delta x$

Step 3. stopping criterion 충족되면 stop

<br>

## 6-2. Gradient Descent in (non-) convex functions

1. convex function
   - local minimum = global minimum
2. non-convex function
   - local minimum = global minimum은 보장 안됨

![figure2](/assets/img/co/img25.png)

![figure2](/assets/img/co/img26.png)

<br>

## 6-3. Gradient Descent Interpretation

Gradient Descent가 담고 있는 의미

- **$f$ 를 2차로 근사한 이후, 함수의 최소 위치를 다음 위치로 선택**

<br>

**Taylor Series Expansion (2차까지)**

$f(y) \approx f(x)+\nabla f(x)^{T}(y-x)+\frac{1}{2} \nabla^{2} f(x)\|y-x\|_{2}^{2}$.

- 위에서, hessian $\nabla^{2} f(x)$ 를 $\frac{1}{t} I$ 로 대체하면…

  - $f(y) \approx f(x)+\nabla f(x)^{T}(y-x)+\frac{1}{2 t}\|y-x\|_{2}^{2}$.

  - hessian을 계산하기엔 computationally expensive..
  - 여기서 $t$ 는 step size

<br>

요약 : ***”step size의 역수가 eigen-value인 Hessian”을 2차 항의 계수로 가지는 2차식으로 근사***

<br>

- 선형 근사 : $f(x)+\nabla f(x)^{T}(y-x)$

- proximiity term : $\frac{1}{2 t}\|y-x\|_{2}^{2}$

<br>

위와 같이 함수를 2차식으로 근사한다.

이 2차식을 최소화 하는 지점을, 다음 지점을 선택한다. ( $f(y)$의 gradient가 0이 되는 지점 )

- 다음 지점 : $y = x^{+}$

$\rightarrow $ $x^{+}=x-t \nabla f(x)$.

<br>

![figure2](/assets/img/co/img27.png)

<br>

## 6-4. Step Size 고르기

GD를 할 때, step size에 따라 속도 / 발산 issue가 생긴다.

적절한 step size는 어떻게 찾을까?

<br>

### a) Fixed step size

가장 단순한 방법 ( 어떠한 step에서든 동일한 $t$ 사용 )

<br>

$t$가 너무 커도, 너무 작아도 문제다!

example ) $f(x)=\left(10 x_{1}^{2}+x_{2}^{2}\right) / 2$

![figure2](/assets/img/co/img28.png)

<br>

### b) Backtracking line search

Fixed step size의 문제점

- 경사가 평평한 구간에서는, optimal point를 지나쳐서 진동할 수도
- 경사가 가파른 구간에서는, 진행이 느릴 수도

$\rightarrow$ 곡면의 특성에따라 step size를 adaptive하게 조절해야!

이 중 하나가 **backtracking line search**

<br>

![figure2](/assets/img/co/img29.png)

- 아래쪽 점선 : 접선

- 위쪽 점선 : 접선의 기울기에 $\alpha$를 곱한 방향으로 한 step을 간 경우

<br>

직관적 idea : 한 step을 간 지점에서, $f(x+t \Delta x)$ 가

- $f(x)+\alpha t \nabla f(x)^{T} \Delta x$ 위에 있으면 : 너무 큰 step size
- $f(x)+\alpha t \nabla f(x)^{T} \Delta x$ 아래에 있으면 : 적당한 step size

<br>

### Process

사용하는 파라미터 : $\alpha, \beta$

notation : $\Delta x=-\nabla f(x)$

- step 1) initialize $\alpha, \beta$
  - $0<\beta<1$ , $0< \alpha \leq 1/2$

- step 2) initialize $t$ 
  - $t = t_{init} = 1$
- step 3) 아래를 반복
  - $f(x-t \nabla f(x))>f(x)-\alpha t\|\nabla f(x)\|_{2}^{2}$ 일 경우, $t=\beta t$ 로 줄이기
  - 위 조건이 만족되지 않을때 까지 반복
- step 4) gradient descent
  - $x^{+}=x-t \nabla f(x)$.
- stopping criterion을 만족할 때 까지 step 2 ~ step 4 반복

<br>

### Summary

- simple, but WORKS WELL!
- $\alpha$ : 다음 step의 방향을 결정
- $\beta$ : step을 얼마나 되돌아올지 결정
  - $\beta$ 작을 경우 :
    - 반복은 3번만 하면 됨. ( 조건 금방 충족 )
    - but step size가 작아, 멀ㄹ리 못감
- 대부분 $\alpha = 1/2, \beta \approx 1$ 로 설정

<br>

위의 process의 step3의 조건 :

![figure2](/assets/img/co/img30.png)

![figure2](/assets/img/co/img31.png)

<br>

### c) Exact line search

- GD의 곡면의 특성에 맞춰 step size를 adaptive 하게 설정하는 방법 중 하나

- 한 줄 요약 : negative gradient의 직선을 따라, 가장 좋은 step size 설정
- 수식 : $t=\operatorname{argmin}_{s \geq 0} f(x-s \nabla f(x))$.
- 단점 : 실용적 X
  - step size를 exhaustive하게 탐색

<br>

## 6-3. Convergence Analysis

pass

<br>

## 6-4. Gradient Boosting

https://seunghan96.github.io/ml/ppt/3.Boosting/ 참고

<br>

### a) Gradient Boosting 이란

- 여러 tree를 ensemble
- GD를 사용하여, **순차적**으로 tree 생성
  - 다음 tree : 이전 tree의 오차를 보완하는 방식으로!
- 회귀 & 분류 모두 OK

<br>

### b) Functional gradient descent

- 함수 공간에 대해서 loss function을 최적화
- gradient의 음수 방향을 가지는 함수를 반복적으로 선택

<br>

### c) Gradient Boosting ( detail )

Notation

- \[x\] $x_i, i=1 \cdots n$
- \[y_true\] $y_i, i=1 \cdots n$
- \[y_pred\] $u_i, i=1 \cdots n$

<br>

Prediction

- weighted average of $M$ Trees

  ( = $u_{i}=\sum_{j=1}^{M} \beta_{j} T_{j}\left(x_{i}\right)$ )

<br>

Loss Function 

- $\min _{\beta} \sum_{i=1}^{n} L\left(y_{i}, \sum_{j=1}^{M} \beta_{j} T_{j}\left(x_{i}\right)\right)$.

<br>

<br>

Optimization Problem

- 위의 Loss Function를 최소화하는 $M$ 개의 가중치 $\beta_{j}$ 를 찾는 문제
- 재정의 : $\min _{u} f(u)$
  - where $L(y, u)$

<br>

Gradient Boosting :

- 위의 $\min _{u} f(u)$ 를 gradient descent를 사용해서 풀기

<br>

### d) Algorithm

<br>

## 6-5. Stochastic Gradient Descent

