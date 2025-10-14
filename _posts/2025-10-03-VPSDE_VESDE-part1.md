---
title: VP-SDE (Variance Preserving SDE) - Part 1
categories: [DIFF, MULT, CV]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# **1. [DDPM] VP-SDE (Variance Preserving SDE)**

## (1) Foward SDE

$$\boxed{dx = -\tfrac{1}{2}\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dw}$$.

- Noise schedule $$\beta(t) = 1-\alpha(t)$$ :

  - 초기에 작게 시작 

  - 주로 $$\alpha(0)=1$$로 시작함 (== $$\beta(0)=0$$)

<br>

## (2) 해

$$\boxed{x(t) = \alpha(t) \, x(0) + \sigma(t)\,z,\quad z \sim \mathcal{N}(0,I)}$$.

- $$\alpha(t) = \exp\!\left(-\tfrac{1}{2}\int_0^t \beta(s)\,ds\right)$$.

  (위의 Noise schedule의 $$\alpha(t)$$와는 다른 것)

- $$\sigma^2(t) = 1 - \alpha^2(t)$$.

<br>

## (3) **분산**

$$\boxed{\mathrm{Var}[x(t)] = \alpha^2(t)\,\mathrm{Var}[x(0)] + \sigma^2(t)}$$.

- 위의 **(2) 해**에서 바로 도출 가능함.
- 데이터가 단위 분산(Var=1) 가정 하에:
  - $$\mathrm{Var}[x(t)] = \alpha^2(t) \cdot 1 + (1-\alpha^2(t)) = 1$$.


$$\rightarrow$$ ***시간이 지나도 분산은 항상 일정 (1)*** (데이터 분산을 보존)

<br>

# **2. [SBM] VE-SDE (Variance Exploding SDE)**

## (1) Forward SDE

$$\boxed{dx = \sqrt{\tfrac{d\sigma^2(t)}{dt}}\,dw}$$.

<br>

## (2) 해

$$\boxed{x(t) = x(0) + \sigma(t)\,z,\quad z \sim \mathcal{N}(0,I)}$$.

<br>

## (3) 분산

$$\boxed{\mathrm{Var}[x(t)] = \mathrm{Var}[x(0)] + \sigma^2(t)}$$.

- 위의 **(2) 해**에서 바로 도출 가능함.
- $$\sigma(t)$$는 t에 따라 점점 커지도록 정의되어 있음

$$\rightarrow$$ ***시간이 지날수록 분산은 커짐***

<br>

# 3. Summary

| **과정**   | **해**                              | **분산**                                        |
| ---------- | ----------------------------------- | ----------------------------------------------- |
| **VP-SDE** | $$x(t) = \alpha(t)x(0)+\sigma(t)z$$ | 항상 일정 (예: 1)                               |
| **VE-SDE** | $$x(t) = x(0)+\sigma(t)z$$          | $$\mathrm{Var}[x(0)] + \sigma^2(t)$$, 계속 증가 |

- [DDPM/VP-SDE] ***“분산은 항상 보존, 노이즈로 점점 "교체"되는 과정”***

- [Score-based/VE-SDE]는 ***“분산이 점점 커져서 원래 신호가 노이즈에 묻히는 과정”***

<br>

# 4. 적분인자 (integrating factor)

**(1) 곱의 미분**

$$\boxed{\frac{d}{dt}\big[f(t)g(t)\big]=f’(t)g(t)+f(t)g’(t)}$$.

<br>

**(2) 1차 선형 미분방정식**

$$\boxed{\frac{dx}{dt} + p(t)\,x = q(t)}$$.

<br>

**(3) 적분 인자**

- **선형 미분방정식**을 풀 때 쓰는 고전적인 기법
  
  - How? (선형 미분방정식을 풀 때) **특별한 함수를 선택**해서 곱해줌!
  
  $$\rightarrow$$ 위 식을 직접 풀기는 어려움. 

- 하지만, $$x$$에 ***어떤 함수를 곱해주면*** ...

  $$\rightarrow$$ 좌변이 한 번에 미분 꼴이 되도록 만들 수 있음 (feat. 곱의 미분)

- 그 곱해주는 함수가 바로 **적분인자 integrating factor** ($$I(t)$$)
  - **“적분인자로 잡는다”** = $$I’(t)=p(t)I(t)$$가 되도록 $$I(t$$)를 **내가 골라 곱한다**!

<br>

**(3) 정의: $$I(t)$$**

$$\boxed{I(t) = \exp\!\Big(\int p(t)\,dt\Big)}$$.

이걸 1차 선형 미분방정식에 곱하면 ....

- [Before] $$\frac{dx}{dt} + p(t)\,x = q(t)$$.

- [After] $$I(t)\frac{dx}{dt} + I(t)p(t)x = I(t)q(t)$$,

<br>

[After]의 좌변: $$\frac{d}{dt}\big(I(t)x\big)$$ 꼴로 묶임!

- Step 1) 곱의 미분 법칙을 쓰면,
  - $$\frac{d}{dt}\big(I(t)x(t)\big) = I(t)\frac{dx}{dt} + I’(t)x(t)$$.

- Step 2) 그리고 $$I’(t) = p(t)I(t)$$로 정의했으니,
  - $$\frac{d}{dt}(I(t)x(t)) = I(t)\frac{dx}{dt} + I(t)p(t)x(t)$$.

- Step 3) 결론:
  - $$I(t)\frac{dx}{dt} + I(t)p(t)x(t) = \frac{d}{dt}(I(t)x(t))$$.

<br>

# 5. Itô 미분

## (1) 일반 미분

$$df = f’(t)\,dt$$.

- 변화율이 $$dt$$에 비례해서 작아짐

<br>

## (2) 브라운 운동 $$W_t$$

**브라운 운동(Brownian motion)**

- 매 순간 무작위로 튀는 경로

- 성질: $$W_{t+dt}-W_t \sim \mathcal{N}(0,dt)$$
  - 즉, 평균 $$0$$, 분산 $$dt$$의 정규분포
- 아주 작은 증분은 보통 $$dW_t$$로 적음

<br>

## (3) Itô 미분의 핵심

Brownian motion은 **거칠어서** 고전적 미분이 불가능 (미분 계수 없음)

$$\rightarrow$$ 대신 **Itô calculus**라는 규칙을 사용

<br>

핵심 규칙: $$(dW_t)^2 = dt$$

- $$dt^2 = 0, \quad dt\,dW_t = 0$$.

$$\rightarrow$$ 즉, 브라운 운동의 제곱 변화량은 시간 $$dt$$ 크기의 deterministic term!

<br>

## (4) Itô 미분 공식 (Itô’s lemma)

확률 과정 $$X_t$$가

- $$dX_t = a(X_t,t)\,dt + b(X_t,t)\,dW_t$$

를 따른다면, 

<br>

어떤 함수 $$f(X_t,t)$$의 미분은

- $$df = \Big(\frac{\partial f}{\partial t} •	a\frac{\partial f}{\partial x} •	\tfrac12 b^2 \frac{\partial^2 f}{\partial x^2}\Big)\,dt •	b\frac{\partial f}{\partial x}\,dW_t$$.

<br>

## (5) Itô 곱셈법칙

(일반 product rule과 달리) $$dW$$항이 있어 ***교차항이 존재***한다!!

$$\boxed{d(X_t Y_t) = X_t\,dY_t + Y_t\,dX_t + d\langle X,Y\rangle_t}$$

- $$d\langle X,Y\rangle_t$$: 두 과정의 **이차 변동 (quadratic variation)**

<br>

특히, $$X=Y=W$$일 때

- $$d(W_t^2) = 2W_t\,dW_t + dt$$.

$$\rightarrow$$ 이 추가 $$dt$$ 항이 바로 Itô calculus의 핵심 차이!

<br>

## (6) 직관

- **고전 미분**: 곱셈/연쇄법칙만 있으면 됨.

- **Itô 미분**: 브라운 운동 O

  $$\rightarrow$$  **제곱 변화량이 무시되지 않고** dt**로 남는다**

  $$\rightarrow$$ 이 때문에 SDE 해석에선 반드시 Itô 규칙을 써야
