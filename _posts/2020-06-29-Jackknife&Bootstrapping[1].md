---
title: Jackknife & Bootstrapping [1]
categories: [ML,STAT]
tags: [Machine Learning, Replication variance estimation,Jackknife, Bootstrapping]
excerpt: Replication Variance Estimation
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Jackknife & Bootstrapping [1]

참고 자료 : 임종호 교수님 수리통계학2 강의 자료



## 1. Introduction

데이터가 어떠한 분포에서 나왔는지 모를 때, 어떻게 해당 분포를 추정할 수 있을 것인가?

( 혹은, 어떻게 하면 더 나은 estimator를 세울 수 있을까? )



### 1) Jackknife

- Quenoullie (1949) : 
  - n개의 데이터 $$x_1 ... x_n$$ 중, 하나의 $$x_i$$  만 제외시키고 평균을 구함
  - 이 과정을 총 n번 반복하여 n개의 평균을 구함 ( 각각 $$x_1$$, $$x_2$$ ... 를 제외시키고 평균 냄 )
  - Estimator의 Bias를 줄일 수 있음
    
- Tukey (1958) :
  - variance estimation에도 사용 가능!



### 2) Bootstrapping

- Hartigan (1969) :
  - Jackknife의 개선(확장) 버전
  - $$n$$개의 데이터에서, 총 $$2^n -1$$개의 subset을 만들 수 있음
    
- Efron (1979) :
  - Generalized Resampling technique
  - SRS (Simple Random Sampling) with replacement ( 복원 추출 )



### Keypoint

- **"Variance Estimation"**에 초점
- Replication Variance Estimation 
- 조건 : Random Sample이어야 함

<br>

<br>



## 2. Jackknife Estimator

### 1) Quenoille's Jackknife Estimator

- $$X_1,...X_n$$ : $$f(x ; \theta)$$에서 뽑힌 **random sample**

- $$T_n = T_n(X_1,...,X_n)$$ : $$\theta$$의 estimator

- $$T_{n-1,i} = T_{n-1}(X_1,...,X_{i-1},X_{i+1},....,X_n)$$

  - ( 하나의 X만을 제외하고 추정한 estimator이다 )

- $$T_n$$의 bias  : Bias($$T_n$$) = $$E(T_n) - \theta$$

  



Jackknife Bias Estimator는 다음과 같의 정의한다.

$$b_{jack} = (n-1)(\bar{T}_n - T_n)$$,

where $$T_n = \frac{1}{n}\sum_{i=1}^{n}T_{n-1,i}$$



Jackkinfe Estimator는 (Jackknife Bias Estimator를 사용하여) 다음과 같이 나타낼 수 있다.

$$T_{jack} = T_n - b_{jack} = nT_n - (n-1)\bar{T}$$



결론부터 이야기하자면, 위의 estimator $$T_{jack}$$는 $$T_n$$ 보다 bias가 낮은 estimator이다.

왜 그런지 알아보자.



우선, $$T_n$$의 bias를 다음과 같이 설정하자.

$$Bias(T_{n}) = \frac{a}{n}  + \frac{b}{n^2} + O(\frac{1}{n^3}) = O(\frac{1}{n})$$



따라서, 우리는 $$T_{n-1,i}$$와 $$\bar{T}_n$$의 bias를 다음과 같이 나타낼 수 있다.

$$Bias(T_{n-1,i}) = \frac{a}{n-1}  + \frac{b}{(n-1)^2} + O(\frac{1}{(n-1)^3})$$

$$Bias(\bar{T}_n) = \frac{a}{n-1}  + \frac{b}{(n-1)^2} + O(\frac{1}{(n-1)^3})$$



이를 사용하여, $$E[b_{jack}]$$를 다음과 같이 정리할 수 있다.

$$\begin{align*}
   E[b_{jack}] &=E[(n-1)(\bar{T}-T_n)]\\
   &= (n-1)E[\bar{T}-T_n]\\   &= (n-1)\{(\bar{T}+Bias(\bar{T}_n))-(\bar{T}+Bias(T_n))\}\\ &= (n-1)\{Bias(\bar{T}_n)-Bias(T_n)\}\\&=(n-1)\{(\frac{1}{n-1}-\frac{1}{n})a +(\frac{1}{(n-1)^2}-\frac{1}{n^2})b\} + O(\frac{1}{n^2})\\ &=\frac{a}{n} + \frac{(2n-1)b}{n^2(n-1)} + O(\frac{1}{n^2})
\end{align*}$$



따라서, $$T_{jack}$$의 bias estimator인 $$Bias(T_{jack})$$ 를 구하면 다음과 같다.

$$\begin{align*}
   Bias(T_{jack}) &= Bias(T_n) - E[b_{jack}]\\
   &= -\frac{b}{n(n-1)} + O(\frac{1}{n^2}) \\ &= O(\frac{1}{n^2})
\end{align*}$$

결론 : 

- $$Bias(T_{n}) = O(\frac{1}{n})$$
- $$Bias(T_{jack}) = O(\frac{1}{n^2})$$



따라서, Jackknife를 통해 **bias-reduced** estimator를 구할 수 있다.



### 2) Deleted-d Jackknife Estimator

위의 Quenoille's Jackknife Estimator에서는 하나의 sample만을 제거한 것으로 estimator들을 구하였다면, Deleted-d Jackknife Estimator에서는 다음과 같이 $$d$$개의 sample을 제거한다.



$$T_{r,s} = T_r(X_i, i \in A^{c})$$

where..

- **A** : subset of {1,...,n} with size d
- **r = n-d**



### 3) Tukey의 Deleted-one Jackknife Variance Estimator

Tukey는 Jackknife estimator를 variance estimation을 위해 사용하였다.

$$\widetilde{T}_{n,i} = nT_n - (n-1)T_{n-1,i}$$

- $$\widetilde{T}_{n,i}$$를 iid sample로 취급
- $$\widetilde{T}_{n,i}$$는 $$\sqrt{n}T_n$$과 approximately same variance를 가짐



$$\begin{align*}
V_{jack} &= \frac{1}{n}\sum_{i=1}^{n}\frac{(\widetilde{T}_{n,i} - \frac{1}{n}\sum_{j=1}^{n}\widetilde{T}_{n,j})^2}{n-1}\\
&= \frac{1}{n(n-1)}\sum_{i=1}^{n}(\widetilde{T}_{n,i} - \frac{1}{n}\sum_{j=1}^{n}\widetilde{T}_{n,j})^2\\
&= \frac{1}{n(n-1)}\sum_{i=1}^{n}((nT_n - (n-1)T_{n-1,i}) - \frac{1}{n}\sum_{j=1}^{n}(nT_n - (n-1)T_{n-1,j}))^2\\
&= \frac{n-1}{n}\sum_{i=1}^{n}(T_{n-1,i}-\frac{1}{n}\sum_{j=1}^{n}T_{n-1,j})^2
\end{align*}$$



### Remarks

우리는 지금까지 1~d개의 sample을 제외 시키고 estimator를 추정하였다. 하지만 이를, '제거'가 아닌 '다른 가중치(weight)'를 주었다는 관점으로도 볼 수 있다.

예를 들면, $$\bar{X}_{n-1,i}$$는 , $$X_i$$를 제외한 나머지 sample들로 계산한 평균으로 볼 수도 있지만, 다르게 보면 다음과 같은 weighted average로 바라볼 수도 있다.

$$\bar{X}_{n-1,i} = \frac{1}{n}\sum_{j=1}^{n}w_jX_j$$

where $$w_i =0 $$ and $$w_j=\frac{n}{n-1}$$ for $$j \neq i$$













