---
title: (08) Duality
categories: [CO]
tags: [Convex Optimization]
excerpt: (참고) 모두를 위한 convex optimization

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# 08. Duality

## 8-1. Lower Bounds in LP

### Example) Constraint들의 선형조합으로 Objective function 표현 

<br>

**(1) Problem**

$$\begin{array}{ll}
\min _{x, y} & x+3 y \\
\text { subject to } & x+y \geq 2 \\
& x, y \geq 0 
\end{array}$$.

<br>

**(2) Constraint들의 선형조합으로 Objective Function 꼴 만들기**

$$\begin{array}{rr} 
& x+y \geq 2 \\ + & 0 x \geq 0 \\ + & 2 y \geq 0 \\
= & x+3 y \geq 2
\end{array}$$.

$$\rightarrow$$ Lower bound $$B=2$$

<br>

### Example) General Form

<br>

**(1) Problem**

$$\begin{array}{ll}
\min _{x, y} & px+qy \\
\text { subject to } & x+y \geq 2 \\
& x, y \geq 0 
\end{array}$$.

<br>

**(2) Constraint들의 선형조합으로 Objective Function 꼴 만들기**

$$\begin{aligned}
&a(x+y) \geq 2 a\\
&+\quad b x \geq 0\\
&+\quad c y \geq 0\\
&=\quad(a+b) x+(a+c) y \geq 2 a
\end{aligned}$$

$$\rightarrow$$ Lower bound $$B=2a$$

<br>

where, 아래 조건 충족!

- (1) $$a+b=p$$
- (2) $$a+c=q$$
- (3) $$a, b, c \geq 0$$

<br>

**(3) Dual LP로 변형하기**

- 위 문제 ( = Primal LP )를, 아래와 같이 변형해서 ( = Dual LP ) 풀 수 있다.
- 이 때, 최적화 변수 ( optimization variable )은, Primal LP에서의 “constraint의 개수”와 동일하다.
- 아래의 식에서, dual variable은 $$a,b,c$$ 이다.

$$\begin{array}{ll}
\max _{a, b, c} & 2 a \\
\text { subject to } & a+b=p \\
& a+c=q \\
a, b, c \geq 0
\end{array}$$.

<br>

### Example) Constraint들의 선형조합으로 Objective function 표현 (2)

<br>

**(1) Problem**

$$\begin{array}{ll}
\min _{x, y} & px+qy \\
\text { subject to } & x \geq 0 \\
& y \leq 1 \\ & 3x+2y=2 
\end{array}$$.

<br>

**(2) Constraint들의 선형조합으로 Objective Function 꼴 만들기**

$$\begin{array}{rr} + & a x \geq 0 \\ + & -b y \geq-b \\
= & (a+3 c) x+(-b+c) y \geq 2 c-b
\end{array}$$.

$$\rightarrow$$ Lower bound $$B=2c - b$$

<br>

where, 아래 조건 충족!

- (1) $$a+3c=p$$
- (2) $$-b+c=q$$
- (3) $$a, b \geq 0$$

<br>

**(3) Dual LP로 변형하기**

- 위 문제 ( = Primal LP )를, 아래와 같이 변형해서 ( = Dual LP ) 풀 수 있다.
- 이 때, 최적화 변수 ( optimization variable )은, Primal LP에서의 “constraint의 개수”와 동일하다.

$$\begin{array}{ll}
\max _{a, b, c} & 2c-b \\
\text { subject to } & a+3c=p \\
& -b+c=q \\
a, b \geq 0
\end{array}$$.

<br>

## 8-2. Duality in general LPs

이번엔, matrix form으로 알아볼 것이다.

<br>

**(0) Notation (shape)**

$$c \in \mathbb{R}^{n}, A \in \mathbb{R}^{m \times n}, b \in \mathbb{R}^{m}, G \in \mathbb{R}^{r \times n}, h \in \mathbb{R}^{r}$$.

<br>

**(1) Problem**

$$\begin{array}{ll}
\min _{x} & c^{T} x \\
\text { subject to } & A x=b \\
& G x \leq h
\end{array}$$.

<br>

**(2) Constraint들의 선형조합으로 Objective Function 꼴 만들기**

$$\begin{aligned}
& u^{T}(A x-b)=0 \\
+& v^{T}(G x-h) \leq 0 \\
=& u^{T}(A x-b)+v^{T}(G x-h) \leq 0
\end{aligned}$$.

<br>

위 부등식을 재 정리하면….

( 참고 : $$u$$ 는 “부호 무관”, $$v$$ 는 “양수” )

$$\begin{array}{r}
u^{T}(A x-b)+v^{T}(G x-h) \leq 0 \\
\underbrace{\left(-A^{T} u-G^{T} v\right)^{T}}_{=c^{T}} x \geq-b^{T} u-h^{T} v
\end{array}$$.

$$\rightarrow$$ Lower bound $$B=-b^{T} u-h^{T} v$$

<br>

where, 아래 조건 충족!

- (1) $$c=-A^{T} u-G^{T} v$$
- (2) $$v \geq 0$$

<br>

**(3) Dual LP로 변형하기**

- 위 문제 ( = Primal LP )를, 아래와 같이 변형해서 ( = Dual LP ) 풀 수 있다.
- 이 때, 최적화 변수 ( optimization variable )은, Primal LP에서의 “constraint의 개수”와 동일하다.

$$\begin{array}{ll}
\max _{u,v} & -b^{T} u-h^{T} v \\
\text { subject to } & c=-A^{T} u-G^{T} v \\
& v \geq 0 
\end{array}$$.

<br>



