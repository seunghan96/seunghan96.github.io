---
title: Factorization Machine
categories: [ML,STAT]
tags: [Machine Learning, FM, Recommendation System]
excerpt: Factorization Machine
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Factorization Machine

참고 논문 : Factorization Machines ( Steffen Rendle, Department of Reasoning for Intelligence )



## 1. Introduction

### (1) What is Factorization Machine?

FM(Factorization Machine) =  **SVM ( Support Vector Machine )** + **Factorization**

핵심 : **"모든 변수 들 사이의 상호 작용 효과(Interaction Effect)"**를 고려할 수 있다.

장점 : sparse data에서도 잘 작동함 
 ( recommendation system 문제를 잘 풀 수 있음)



FM 외의 다른 Factorization Model들도 있으나 ( 대표적으로, matrix factorization ), 이러한 방법들은 데이터는 일반적인 데이터에 쉽게 적용 불가! general task 풀기에는 FM이 좋다!



### (2) Drawback of SVM

Random Forest가 나오기 전에, 유명했던 SVM! 

SVM의 큰 한계점은, **"sparse data"하에서는 잘 작동하지 않는다**는 점!

( SVM에 관한 내용은 [https://seunghan96.github.io/ml/%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C/2.SVM/](https://seunghan96.github.io/ml/발표자료/2.SVM/) 참고 )



### (3) Advantages of FM

-  1 ) sparse data 하에서도 잘 작동함
-  2 ) linear complexity ( SVM 처럼 support vector에 의존 X )
-  3 ) General Predictor ( 다른 Factorization Model들은 매우 specific한 data에서만 잘 작동했는데, FM은 input data에 크게 영향 받지 않아 )

<br>

<br>

## 2. Sparsity Problem

이 논문은 기본적으로 data가 sparse한 문제인 상황을 다룬다.

( recommendation system이나, BOW (bag-of-words) approach에서 sparse data를 자주 접하게 된다 )

<img src="https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-3-030-25217-5_2/MediaObjects/481651_1_En_2_Fig1_HTML.png" width="550" /> <br>



왜 sparse할까?

일단 기본적으로 sparse한 상황이 발생하는 이유는, **"다양한 Categorical Variable domain"**을 가지기 때문!

- ex ) 영화 추천을 해주려 하는데, 이 세상에 존재하는 영화의 수는 매우 많아!
- ex ) NLP에서, 특정 텍스트 이후에 나오게 될 단어를 예측하려 하는데, 단어의 종류도 매우 많아!



그렇다면 왜 FM은 위와 같은 sparse data의 상황에서 문제를 잘 해결할 수 있을까? 

<br>

<br>

## 3. Factorization Machines (FM)  Models

### (1) Model equation

- 우리가 알고 있던 기존의 선형 모델은 다음과 같은 model equation을 가졌다. ( 총 n개의 변수 )

  $$\hat{y}(x) := w_0 + \sum_{i=1}^{n}w_ix_i$$
  
- 이에 반해, FM은 다음과 같은 model equation을 가진다. ( degree=2의 FM )

  $$\hat{y}(x) := w_0 + \sum_{i=1}^{n}w_ix_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n}<v_i,v_j>x_ix_j$$


  여기서, $$<v_i,v_j>$$는 $$k$$ 차원의 두 벡터의 내적이고, 모든 변수 $$x_i$$는 각각에 해당되는 벡터 $$v_i$$를 가지고 있다.

  $$< v_i, v_j > := \sum_{f=1}^{k}v_{i,f}\cdot v_{j,f}$$

  

위 식을 보면 알 수 있듯, (2-degree) FM은 **모든 변수들 간의 상호작용 효과**를 잡아낼 수 있다!
( ($$x_1,x_2$$), ($$x_1,x_3$$), ($$x_1,x_4$$), ... ($$x_{n-1},x_n$$) )

또한 직관적으로 생각했을 때, 위 식에서 벡터 $$v$$ 의 차원을 너무 크지 않게 해야 generalize하기 좋다는 것도 알 수 있다. 

<br>

## (2) Parameter Estimation Under Sparsity

sparse data에서는 여러 $$x$$변수들 사이의 상호작용 효과를 고려하기에 데이터가 충분하지 않은 경우가 종종 발생할 것이다. 
ex) $$x_{100}$$ 과 $$x_{120}$$ 의 값을 둘 다 가지고 있는 데이터가 없을 경우, $$x_{100}$$과 $$x_{120}$$ 사이의 interaction effect는 잡아낼 수 없을 것이다.



한번 생각해보자.

- $$X_1$$ : A장르의 영화 ( 좋아하면 1, 아니면  0 )
- $$X_2$$ : B장르의 영화 ( 좋아하면 1, 아니면  0 )
- $$X_3$$ : a 음식 ( 좋아하면 1, 아니면  0 )
- $$X_4$$ : b 음식 ( 좋아하면 1, 아니면  0 )



A장르의 영화($$X_1$$)를 좋아하고, a 음식($$X_3$$)을 좋아하는 국현이가 이번에 딸기를 샀다. 

B장르의 영화($$X_3$$)를 좋아하고, b 음식($$X_4$$)을 좋아하는 지우가 이번에 바나나를 샀다. 

그런데 이때 A장르의 영화($$X_1$$)를 좋아하고 b 음식($$X_4$$)을 좋아하는 경률이형이 과일을 사려고 한다. 



우리는 경률이형에게 과일을 추천해주기 위해 연산을 할 때, $$X_1$$과 $$X_4$$의 interaction term을 고려할 수 없다. Training Data에는 단 한번도 $$X_1$$과 $$X_4$$를 동시에 좋아했던 경우가 없었기 때문에, 이 두 $$X_1$$과 $$X_4$$의 interaction term의 parameter ( $$w_{(1,4)}$$ )를 구한 적이 없었다.  ( $$w_{(1,4)}$$는 0일 것이다 )

그렇다고 경률이형에게 그냥 먹고싶은 것을 알아서 찾아 먹으라고 하고 등 돌릴 수는 없다.  우리는 Factorization Machine을 사용해서 경률이형에게 최적의 과일을 찾아줄 수 있다.

HOW? 

**"by breaking the independence of interaction parameters! ( by factorizing them )"**



### Example

<img src="https://d3i71xaburhd42.cloudfront.net/df93596d4ed71d2863532c063c4c693711216abf/2-Figure1-1.png" width="550" /> <br>



$$X$$ 변수

- 파랑 box : "USER"가 누구인지
- 주황 box : 어떤 "MOVIE"를 봤는지
- 노랑 box : 여태까지 user가 모든 영화에 주었던 "RATE"들 

- 초록 box : 가장 마지막으로 본 영화까지의 "MONTH" 수
- 보라 box : 가장 마지막으로 평점을 주었던  "MOVIE"



$$Y$$ 변수

- 가장 마지막으로 본 영화의 "예상 평점"



(파란)$$A$$와 (주황)$$ST$$  사이의 상호작용 효과를 구하고 싶다고 해보자. 하지만, 문제는 $$A$$는 단 한번도 $$ST$$를 본적이 없다. 즉, $$w_{(A,ST)}$$는 0이 될 것이다. 하지만, 우리는 이 둘 사이의 interaction term을 factorize함으로써, (즉, 각각의 X가 자신만의 vector를 가지고, 두 X 사이의 내적인 $$<v_A,v_{ST}>$$ 가 interaction term의 역할을 함 ) 이를 구할 수 있다.

우선 B 친구와 C친구는 둘 다 SW영화에 높은 평점을 주었다는 점에서 비슷한 벡터를 가질 것이다. ( $$v_B$$와 $$v_C$$는 유사할 것이고, $$<v_B,v_{SW}>$$ 와 $$<v_C,v_{SW} >$$ 도 유사할 것이다. )

하지만 A친구와 C친구는 매우 다른 vector를 가질 것이다. ( 이 둘은 각각 T와 SW에 정 반대의 평점을 남겼다. 취향이 달라도 너무 다르다. ) 또한, ST와 SW는 매우 비슷한 vector를 가질 것이다. ( B라는 친구가 ST와 SW에 아예 동일한 평점을 남겼다! )

이를 통해, 우리는 유추할 수 있다. $$<v_A,v_{SW}>$$와 $$<v_A,v_{ST}>$$는 매우 유사할 것이라는 점이다. A는 SW를 본 적이 있기 때문에, 본 적이 없는 ST 영화에 대해서도 (A와 ST 사이의) interaction effect를 유추할 수 있는 것이다!

<br>

### (3) Computation

생각보다 시간 복잡도가 그다지 높지 않다.

FM의 model equation을 생각하면,

 $$\hat{y}(x) := w_0 + \sum_{i=1}^{n}w_ix_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n}x_ix_j$$ 

시간 복잡도는 $$O(k\; n^2)$$라는 것을 알 수 있다.



하지만 우리는 위 식을 Lemma 3.1을 통해, **시간 복잡도를 $$O(k \; n)$$으로 줄일 수 있다!**



**[ Proof ]**

$$\sum_{i=1}^{n}\sum_{j=i+1}^{n}<v_i,v_j>x_i x_j$$

$$ = \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}<v_i,v_j>x_i x_j - \frac{1}{2}\sum_{i=1}^{n}<v_i,v_i>x_i x_i$$

$$ = \frac{1}{2}(\sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{f=1}^{k}v_{i,f},v_{j,f} x_i x_j - \sum_{i=1}^{n}\sum_{f=1}^{k}v_{i,f}v_{i,f}x_i x_i)$$

$$=\frac{1}{2}\sum_{f=1}^{k}((\sum_{i=1}^{n}v_{i,f}x_i)(\sum_{j=1}^{n}v_{j,f}x_j) - \sum_{i=1}^{n}v_{i,f}^2x_i^2)$$

$$=\frac{1}{2}\sum_{f=1}^{k}((\sum_{i=1}^{n}v_{i,f}x_i)^2 - \sum_{i=1}^{n}v_{i,f}^2x_i^2)$$





<br>

<br>

## 4. FM as Predictors

FM은 다양한 prediction task를 풀 수 있다. 대표적으로 다음과 같이 세 가지 문제가 있다.

- 1 ) Regression
  - 위의 model equation에서 $$\hat{y}(x)$$ 
  
- 2 ) Binary Classification
  - sign of $$\hat{y}(x)$$ 
  
- 3 ) Ranking

<br>

<br>

## 5. Learning FM

SGD (Stochastic Gradient Descent)를 통해 효율적으로 학습할 수 있다.

FM 모델의 기울기(gradient)는 다음과 같다.

$$\frac{\partial}{\partial \theta} \hat{y}(x)$$ = $$\left\{\begin{matrix}
1 \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\; if \; \theta =0\\ 
x_i \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\; if \; \theta =w_i\\ 
x_i \sum_{j=1}^{n}v_{j,f}x_j - v_{i,f}x_i^2 \;\;\;\;\;\;\; if \; \theta = v_{i,f}
\end{matrix}\right.$$


( 위 기울기에서 $$\sum_{j=1}^{n}v_{j,f}x_j$$는 $$i$$와 무관하기 때문에, 미리 계산한 뒤 사용할 수 있다. )

<br>

<br>

## 6. d-way FM

지금 까지는 2-degree (2-way)의 FM만을 고려했다. 이를 d-degree로 일반화 하면 다음과 같이 나타낼 수 있다.

$$\hat{y}(x) := w_0 + \sum_{i=1}^{n}w_ix_i + \sum_{l=2}^{d}\sum_{i_1=1}^{n}\cdot \cdot \sum_{i_l = i_{l-1} +1}^{n} (\prod_{j=1}^{l} x_{i_j})(\sum_{f=1}^{k_l} \prod_{j=1}^{l}v_{i_j,f}^{(l)})$$

<br>

FM을 요약하자면, 그다지 복잡한 알고리즘은 아니면서도 **크지 않은 연산량으로도 모든 변수들 간의 상호작용 효과를 고려**할 수 있다는 점에서 좋은 모델이다.



