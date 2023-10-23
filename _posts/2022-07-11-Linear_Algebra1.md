---
title: Active Learning
categories: [ML]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>



선형대수  = 행렬과 벡터에 대한 학문



$\begin{aligned}
& 1 x+2 y=4 \\
& 2 x+5 y=9
\end{aligned}$

$\underbrace{\left[\begin{array}{ll}
1 & 2 \\
2 & 5
\end{array}\right]}_{\text {Matrix }} \underbrace{\left[\begin{array}{l}
x \\
y
\end{array}\right]}_{\text {Vector }}=\underbrace{\left[\begin{array}{l}
4 \\
9
\end{array}\right]}_{\text {Vector }}$



전치 (transpose)

$\begin{aligned}
& \left(A^{\top}\right)^{\top}=A \\
& (A+B)^{\top}=A^{\top}+B^{\top} \\
& (A B)^{\top}=B^{\top} A^{\top}
\end{aligned}$

$\begin{aligned}
& (cA)^{\top}=cA^{\top} \\
& \operatorname{det}\left(A^{\top}\right)=\operatorname{det}(A)
\end{aligned}$

$\left(A^{\top}\right)^{-1}=\left(A^{-1}\right)^{\top}$



norm

$a^{\top} b=\| a \| \cdot\| b\| \cos \theta$

$\begin{aligned}
& =\|a\| \cos \theta\|b\| \\
& =\|b\| \cos \theta\|a\|
\end{aligned}$

- a를 b에 정사영 (projection) 한 크기
- b를 a에 정사영 (projection) 한 크기

$\begin{aligned}
& a^{\top} b=\|a\| \cdot\|b\| \cos \theta \\
& a^{\top} a=\|a\| \cdot\|a\|=\|\underline{a}\|^2
\end{aligned}$



unit vector : 크기가 1인 벡터

- $\frac{a}{\sqrt{a^{\top} a}}$ : $a$ 를 unit vector로 normalize



$a$를 $b$ 에 정사영한 벡터는?

$a^{\top} (\frac{b}{\sqrt{b^{\top} b}}) \cdot \frac{b}{\sqrt{b^{\top} b}}$.

- Step1) a와 b방향의 unit vector랑 내적 (방향)
- Step 2) 해당 값을 b방향의 unit vector랑 내적 (크기)



p-norm



$\|a\|_2=\sqrt{a^{\top}a} =\sqrt{1^2+2^2+3^2} =\left(1^2+2^2+3^2\right)^{\frac{1}{2}}$

$\|b\|_1=|1|+|2|+ |-3|=6$

$\|x\|_p \triangleq\left(\sum_T\left|x_T\right|^p\right)^{\frac{1}{p}}$

$\|\underline{x}\|_{\infty} \triangleq \max _i\left|x_i\right|$  ....... infinity norm



행렬 곱셈의 4가지 관점

1. 내적 

$A B=\left[\begin{array}{ll}
a_1^{\top} \\ a_2^{\top} \\ a_3^{\top}
\end{array}\right]\left[\begin{array}{lll}
b_1 & b_2 & b_3
\end{array}\right]=\left[\begin{array}{ll}
a_1^{\top}b_1 \quad a_1^{\top}b_2 \quad a_1^{\top}b_3\\ 
a_2^{\top}b_1 \quad a_2^{\top}b_2 \quad a_2^{\top}b_3\\ 
a_3^{\top}b_1 \quad a_3^{\top}b_2 \quad a_3^{\top}b_3\\ 
\end{array}\right]$



2. Rank-1 matrix의 합

$A B=\left[\begin{array}{lll}
a_1 & a_2 & a_3
\end{array}\right]\left[\begin{array}{ll}
b_1^{\top} \\ b_2^{\top} \\ b_3^{\top}
\end{array}\right]=a_1b_1^{\top} + a_2b_2^{\top} +a_3b_3^{\top}$



3. Column space

$A x=\left[\begin{array}{lll}
a_1 & a_2 & a_3
\end{array}\right]\left[\begin{array}{l}
x_1 \\
x_2 \\
x_3
\end{array}\right]=a_1 x_1+ a_2 x_2+ a_3 x_3$

- $a_1$에 $x_1$배만큼 스칼라배
- $a_2$에 $x_2$배만큼 스칼라배
- $a_3$에 $x_3$배만큼 스칼라배

이 세개의 벡터가 span하는 공간! ( 3개가 독립일 경우 3차원 표현 가능 )



4. Row space

$x^{\top} A=\left[x_1 \quad x_2 \quad x_3\right] \left[\begin{array}{ll}
a_1^{\top} \\ a_2^{\top} \\ a_3^{\top}
\end{array}\right]=x_1 a_1^{\top}+x_2 a_2^{\top}+x_3 a_3^{\top}$



Linear Combination

- scalar배 한 뒤 더하기

- $a_1v_1 + a_2 v_2 + a_3 v_3$

  - $v_1$을 $a_1$ 만큼, $v_2$를 $a_2$ 만큼 ....

- 3개 사용한다고 해서 3차원 표현 가능?

  ( = 3차원을 span할 수 있나 ? )

  - 항상은 NO! 3개가 linearly independent 해야

<br>

선형 독립 (linearly independent)

- independent한 벡터의 수 = 표현할 수 있는 차원의 수

<br>

기저 (Basis) : 어떤 공간을 이루는 필수적인 구성 요소

<br>

직교 행렬 ( orthogonal matrix ): $Q$

- column vector들 = orthonormal하다
  - column들이 서로 직교
- 반드시 square matrix이다.
- $Q^{\top}Q=I$.
- $Q^{-1}=Q^{T}$.

<br>

Rank ( 행렬의 개수 )

-  행렬이 가지는 independent한 column의 수
- column space의 dimension
- column들이 span할 수 있는 차원의 수
- independent한 column의 수 = independent한 row의 수
  - $rank(A) = rank(A^{\top})$.



Example)

$\left[\begin{array}{lll}
1 & 2 & 3 \\
0 & 0 & 0
\end{array}\right]$.

- rank : 1 
  - (1,0), (2,0), (3,0) -> 전부 (a,0)
  - 1차원 밖에 표현 못함

- rank deficient

<br>

Example)

$\left[\begin{array}{lll}
1 & 0 & 1 \\
0 & 1 & 1
\end{array}\right]$.

- rank : 2
  - (1,0), (0,1), (1,1) -> (a,0), (0,a)
  - (m,n)행렬이면, rank의 최대 값은 min(m,n)
- full row rank

<br>

Example)

- 3x2 행렬 & rank=2 : full column rank
- 3x3 행렬 & rank=3 : full rank
- 3x3 행렬 & rank=2 : rank-deficient

<br>

Null Space (영공간)

- Null = 아무 겂도 없다.
- $Ax=0$을 만족하는 $x$의 집합

<br>

Example

- $A = \left[\begin{array}{lll}
  1 & 0 & 1 \\
  0 & 1 & 1
  \end{array}\right] $.
- $A \underline{x}=x_1\left[\begin{array}{l}
  1 \\
  0
  \end{array}\right]+x_2\left[\begin{array}{l}
  0 \\
  1
  \end{array}\right]+x_3\left[\begin{array}{l}
  1 \\
  1
  \end{array}\right]=\left[\begin{array}{l}
  0 \\
  0
  \end{array}\right]$.
- solution? ( = 영벡터 제외 )
  - $c\left[\begin{array}{c}
    1 \\
    1 \\
    -1
    \end{array}\right]$ 
  - row vector의 차원과 같다.

<br>

A가 mxn일 때, $dim(N(A)) = n-r$

- Rank + null space의 dimension = column의 수
  - null space의 dimension : null space의 벡터들이 span할 수 있는 공간의 차원

Null space는 row space와 수작한 space임

- 수직한다 = 내적 시 0



해의 개수

- ex) $\infty$
  - $\begin{aligned}
    & x+2 y=1 \\
    & 2 x+4 y=2
    \end{aligned}$

- ex) 0개
  - $\begin{aligned}
    & x+2 y=1 \\
    & 2 x+4 y=1
    \end{aligned}$
- ex) 1개
  - $\begin{aligned}
    & x+2 y=1 \\
    & x+4 y=1
    \end{aligned}$



Full column rank : 0개 or 1개

Full row rank : $\infty$

Full rank : 1개

Rank deficient : 0개 or $\infty$



정사각행렬  $A$가 invertible 하다

( = non-singular $A$ )

동치인 것들 :

- $det(A) \neq 0$.

- $A$가 full rank

  ( 즉, $det(A)=0 \leftrightarrow $ $A$는 rank deficient )

- $N(A)=0$.

  - null space에 0 벡터밖에 없음



역행렬 property

- $(A B)^{-1}=B^{-1} A^{-1}$
- $\left(A^{-1}\right)^{-1}=A$.
- $(K A)^{-1}=\frac{1}{K} A$.
- $\left(A^{\top}\right)^{-1}=\left(A^{-1}\right)^{\top}$.
- $det(A^{-1}) = \frac{1}{det(A)}$.



Determinant 관련 properties

- $det(A) = 0 \leftrightarrow $ $A$ is singular
- $det(A) = 0 \leftrightarrow $ $A$ is rank deficient
- triangular matrix의 경우
  - $det(A) = a_{11} a_{22} \cdots a_{nn}$.
- diagonal matrix의 경우
  - $det(A) = a_{11} a_{22} \cdots a_{nn}$.
- $det(I) =1$.
- $det(cA)$ = $c^n det(A)$ ( $A$ = $n \times n$ matrix )
- $det(A) = det(A^{\top})$.
- $det(AB) = det(A)det(B)$.
- $det(A^{-1}) = \frac{1}{det(A)}$.
- $det(A) = \lambda_1 \cdots \lambda_n$.



Trace

$tr(A) = \sum_{i=1}^{n} a_ii$.

- diagonal element들의 합
- 정사각행렬에 대해서만 적용
- 언제 유용함?
  - loss(A)를 minimize하고 싶은데.... A는 행렬, scalar로 표현해야하는데...
- $tr(A+B) = tr(A)+tr(B)$.
- $tr(cA) = ctr(A)$.
- $tr(A^{\top}) = tr(A)$.
- $tr(AB) = tr(BA)$.
- $tr(a^Tb) = tr(ba^T)$.
- $tr(ABCD) = tr(BCDA) = tr(CDAB) = tr(DABC)$.
  - cyclic property
- $tr(A) = \sum_{i=1}^{n}\lambda_i$.



Least Squares (최 소자승법)

- $Ax=b$ 를 풀고 싶음

- 해가 없을수도! 에러 계산 후 , 에라 최소화하자

- $e = b-Ax$ : 에러 벡터

  - 수직일 때 에러가 최소가 됨

- $(b-A \hat{x})^{\top} A x=0$.

  $\left(b^{\top} A-\hat{x}^{\top} A^{\top} A\right) \hat{x}=0$.

  $b^{\top} A=x^{\top} A^{\top} A$.

  $A^{\top} b=A^{\top} A \hat{x}$. $\rightarrow$ "normal equation"

- $rank(A^TA) = rank(A)$

- $\hat{x} = (A^{\top}A)^{-1}A^{\top}b$.
- $A\hat{x} = A(A^{\top}A)^{-1}A^{\top}b$.
  - $A(A^{\top}A)^{-1}A^{\top}$ = "projection matrix"



https://www.youtube.com/watch?v=g0eaDeVRdZk&list=PL_iJu012NOxdZDxoGsYidMf2_bERIQaP0&index=8

# Reference

https://www.youtube.com/watch?v=7vV2SF8DyQE&list=PL_iJu012NOxdZDxoGsYidMf2_bERIQaP0