---
title: Hidden Markov Model
categories: [STAT]
tags: [Bayesian,Statistics]
excerpt: Hidden Markov Model (HMM)
---

# Hidden Markov Model

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( https://github.com/aailabkaist/Introduction-to-Artificial-Intelligence-Machine-Learning/blob/master/Week09/IE661-Edu3.0-Week%209-icmoon-ver-2.pdf 를 참고하여 공부하고 정리하였습니다 )

## 1. Introduction

이번 포스트에서는 Hidden Markov Model에 대해서 알아볼 것이다. ( 모든 내용은 Markov Chain에 대해 알고 있다는 가정 하에서 작성되었다 )

간단하게 이야기하자면, Hidden Markov Model은 기존의 Markov Model에 **time**이 추가된 것이라고 보면 된다.

<img src="https://i.stack.imgur.com/94axA.png" width="650" /> 

https://i.stack.imgur.com/94axA.png



위 그림에서 **$$z$$는 latent variable**을, **$$x$$는 observed variable(data)**라고 할 수 있다. 기존에 배웠던 Markov Model에서는, 하나의 $$z$$가 여러 개의 $$x$$에 영향을 미쳤었다. 하지만 Hidden Markov Model(이하 HMM)에서는, 시간이 흐름에 따라 $$z$$가 변화하고, 각각의 $$z$$가 $$x$$값을 만든다. 여기서 $$x$$와 $$z$$는 각각 이산형일수도, 연속형일 수도 있다. (하지만 이번 포스트에서는 둘 다 **이산형**인 경우에만 고려할 것이다). 더 나아가기 위해, 알아 두어야 할 용어가 있다.



**1) Initial State Probability**

이것은 처음의 $$z_1$$값이 어떠한 값을 가질지 정해주는 확률이다.  식으로 나타내면 다음과 같다.

$$p(z_1) \sim Mult(\pi_1,..\pi_k)$$

<br>

**2) Transition Probability** ( = $$a$$ )

말 그대로 옮겨갈(Transit) 확률이다. $$z_{t-1}$$에서 $$z_t$$로 변화할 확률을 이야기하는 것으로, 다음과 같이 표시할 수 있다. ( $$z$$는 $$k$$개의 요소를 가진 vector라고하자 )

$$p(z_t \mid z^i_{t-1} =1) \sim Mult(a_{i,1},...a_{i,k})$$



다르게 표현하면, 다음과 같다.

$$p(z_t^j=1 \mid z_{t-1}^i =1) = a_{i,j}$$

<br>

**3) Emission Probability** ( = $$b$$ )

각각의 $$z$$에서 특정 $$x$$를 만들어낼 확률을 이야기한다.

$$p(x_t \mid z_t^i =1) \sim Mult(b_{i,1},...,b_{i,m}) \sim f(x_t \mid \theta_i)$$



다르게 표현하면, 다음과 같다.

$$p(x_t^j=1 \mid z_t^i=1) = b_{i,j}$$

<br>

## 2. Main Problems to solve

HMM을 통해서 해결하려는 문제는 크게 다음과 같이 세 가지로 나뉠 수 있다.

( 1. Evaluation question, 2. Decoding question, 3. Learning question )

### 1 ) Evaluation Question

- 주어진 정보 : $$\pi, a,b,X$$
- 구하고자 하는 정보 : $$p(X \mid M, \pi, a,b)$$  ( 여기서 M은 HMM의 구조를 뜻한다 )

- **"주어진 모델에서, $$X$$가 관측될 확률은 얼마인가?"**



### 2 ) Decoding Question

- 주어진 정보 : $$\pi, a,b,X$$
- 구하고자 하는 정보 : $$p(Z \mid X,M, \pi, a,b)$$  

- **"가장 그럴듯한 (probable) latent variable $$Z$$ 는 무엇일까?"**



### 3 ) Learning Question

- 주어진 정보 : $$X$$
- 구하고자 하는 정보 : $$\underset{\pi,a,b}{argmax}\;p(X \mid M, \pi, a,b)$$  

- **"데이터가 주어졌을 때, HMM의 underlying parameter들은 어떻게 될까?**
- 어떻게 보면 가장 많이 접하게 될 문제 중 하나일 것이다. 우리에게 보이는 정보는 observed variable $$X$$밖에 없는 경우가 대부분이기 때문이다.

<br>

## 3. Finding parameters of HMM

우리는 marginal probability를 이용하여 HMM의 parameter들을 찾을 것이다.

$$p(X \mid \pi,a,b) = \sum_Z p(X,Z \mid \pi,a,b)$$

<br>

우리는 $$p(X)$$를 다음과 같이 나눠서 표현할 수 있다.

$$\begin{align*}
p(X) &= \sum_zp(X,Z) \\
&= \sum_{z_1}..\sum_{z_t}p(x_1,...x_t,z_1,...z_t)\\
&= \sum_{z_1}..\sum_{z_t}\pi_{z_1}\prod_{t=2}^{T}a_{z_{t-1},z_{t}}\prod_{t=1}^{T}b_{z_t,x_t}
\end{align*}$$



하지만 위와 같이 연산하게 될 경우, 반복적인 계산이 매우 많아진다. 이를 해결하기 위해, dynamic programming을 이용할 것이다. 우선 repeating structure를 위 식에서 이끌어내보자.

$$\begin{align*}
p(x_1,...,x_t,z_t^k=1) &= \sum_{z_{t-1}}p(x_1,...x_{t-1},x_t,z_{t-1},z_t^k=1)\\
&= \sum_{z_{t-1}}p(x_1,...x_{t-1},x_t,z_{t-1})p(z_t^k=1 \mid x_1,...,x_{t-1},z_{t-1})p(x_t\mid z_t^k=1,x_1,...,x_{t-1},z_{t-1})\\
&= \sum_{z_{t-1}}p(x_1,...x_{t-1},x_t,z_{t-1})p(z_t^k=1 \mid z_{t-1})p(x_t\mid z_t^k=1)\\
&= p(x_t \mid z_t^k=1) \sum_{z_{t-1}}p(x_1,...,x_{t-1},z_{t-1})p(z_t^k=1 \mid z_{t-1})\\
&= b_{z_t^k,x_t}\sum_{z_{t-1}}p(x_1,...,x_{t-1},z_{t-1})a_{z_{t-1},z_t^k}
\end{align*}$$



우리는 위 식에서 다음과 같은 반복적인 term을 발견할 수 있다 ( $$\alpha$$라고 하자 )

$$p(x_1,...,x_t,z_t^k=1) = \alpha_t^k = b_{k,x_t}\sum_{i}\alpha_{t-1}^i a_{i,k}$$

<br>

## 4. Dynamic Programming & Memoization

Dynamic Programming에 관한 내용은 ( https://github.com/seunghan96/datascience/blob/master/Data_Structure/2.Algorithm/4.Dynamic_Programming%26Divide_and_Conquer.ipynb 참고 )

Dynamic Programming은 한마디로 표현하자면 **"하나의 큰 문제를 해결하기 위해, 여러 개의 작은 문제로 나누어서 푼다"**라고 할 수 있다. 똑같은 연산을 반복하지 않기 위해, 나누어진 작은 문제의 해를 저장한 뒤, 이를 합쳐서 큰 문제를 푸는 것이다. 상향식 접근법이라고 할 수 있다.

많이들 알고 있는 "피보나치 수열"이 이 Dynamic Programming으로 풀 수 있는 대표적인 예시이다.

아래 그림으로 확인을 해보자. 제일 위의 Fib5를 계산하기 위해, 모든 node를 계속 계산할 필요 없이, 각 값들 (Fib0~Fib4)까지의 값들만 저장하고 있으면 된다.

<img src="https://www.codesdope.com/staticroot/images/algorithm/dynamic4.png" width="550" /> <br>

https://www.codesdope.com/staticroot/images/algorithm/dynamic4.png

<br>

## 5. Forward & Backward Probability Calculation

### 1 ) Forward Probability Calculation

아무튼, 다음의 식을 우리는 Dynamic Programming 기법을 통해서 풀겠다는 것을 확인했다.

$$p(x_1,...,x_t,z_t^k=1) = \alpha_t^k = b_{k,x_t}\sum_{i}\alpha_{t-1}^i a_{i,k}$$



우리는 위 식에서 $$\alpha_t^k$$를 알아야 한다. 이를 구하면, 우리는 곧 $$p(X)$$를 구할 수 있다. ( Evaluation Question을 풀게 되는 것이다 ) 



### Algorithm

1 ) 그러기 위해, 우선 $$a_1^k$$를 initialize한다

$$\alpha_1^k$$ = $$b_{k,x_1} \pi_k$$

<br>

2 ) 시점 T까지 다음을 반복한다 

$$\alpha_t^k = b_{k,x_t} \sum_i \alpha_{t-1}^i a_{i,k}$$

<br>

3) $$\sum_i \alpha_T^i$$를 반환한다.



이 과정에서, 우리는 Dynamic Programming을 사용하는 것이다. 각 iteration에서 구하게  되는 $$\alpha_t^k$$의 값을 memoization table에 저장한다.



하지만 이 방법 (Forward Probability Calculation)에도 한계점은 있다. 이것은 $$p(x_1,...,x_t,z_t^k=1)$$를 구하기 위해, **t 시점 이전의 X**만을 고려한다는 점이다 ( 모든 시점의 X 고려하지 않음 ) 그래서 나온 방법이 **Backward Probability Calculation**이다.

<br>

## 2 ) Backward Probability Calculation

앞서 말했 듯, 우리는 $$p(x_1,...,x_t,z_t^k=1)$$ 대신, 전체 시점을 고려하여 계산한 $$p(z_t^k=1 \mid X)$$를 구하고 싶다. 이는 마찬가지로 정리하면 다음과 같은 recursive structure로 나오게 되어, 쉽게 구할 수 있다.

$$\begin{align*}
p(z_t^k=1,X) &= p(x_1,...,x_t,z_t^k=1,x_{t+1},...,x_{T})\\
&= p(x_1,...,x_t,z_t^k=1)p(x_{t+1},...,x_{T} \mid x_1,...,x_t,z_t^k=1)\\
&= p(x_1,...,x_t,z_t^k=1)p(x_{t+1},...,x_{T} \mid z_t^k=1)\\
\end{align*}$$



위 식의 $$p(x_1,...,x_t,z_t^k=1)p(x_{t+1},...,x_{T} \mid z_t^k=1)$$에서, 앞 부분은 우리가 이미 구했었다 ( $$\alpha_t^k$$ ) 우리는 그 뒷 부분( $$\beta_t^k$$ )만 구하면 된다.

- $$p(x_1,...,x_t,z_t^k=1)$$ = $$\alpha_t^k$$
- $$p(x_{t+1},...,x_{T} \mid z_t^k=1)$$ = $$\beta_t^k$$

<br>

$$\beta_t^k$$는 다음과 같이 구할 수 있다.

$$\begin{align*}
p(x_{t+1},...,x_{T} \mid z_t^k=1) &= \sum_{z_{t+1}}p(z_{t+1},x_{t+1},...,x_{T} \mid z_t^k=1)\\
&= \sum_i p(z_{t+1}^i=1 \mid z_t^k=1)p(x_{t+1}\mid z_{t+1}^i=1,z_t^k=1)p(x_{t+2},...x_T \mid x_{t+1},z_{t+1}^i=1,z_t^k=1)\\
&= \sum_i p(z_{t+1}^i=1 \mid z_t^k=1)p(x_{t+1}\mid z_{t+1}^i=1)p(x_{t+2},...x_T \mid z_{t+1}^i=1)\\
&= \sum_i a_{k,i}b_{i,x_t}\beta_{t+1}^i
\end{align*}$$

<br>

따라서, 우리는 $$p(z_t^k=1,X)$$를 다음과 같이 간단하게 recursive 구조로 표현할 수 있다.

$$p(z_t^k=1,X) = \alpha_t^k \beta_t^k = (b_{k,x_t}\sum_i \alpha_{t-1}^i a_{i,k})\times(\sum_i a_{k,i}b_{i,x_t}\beta^i_{t+1})$$



이제 모든 것이 완벽한 것처럼 보인다. 모든 X를 고려했을 뿐만 아니라 recursive structure로 표현함으로써 Dynamic Programming을 통해서도 풀 수 있게 만들었다. 하지만, 이 또한 한계점이 있다. 우리는 위 식을 통해 $$t$$시점의 $$z$$만 구할 뿐, 모든 $$z$$값을 가진 $$Z$$ 벡터를 구한 것은 아니다. 이 $$Z$$를 구하기 위해, 우리는 Viterbi Decoding이라는 것을 할 것이다.

<br>

## 6. Viterbi Decoding

Viterbi Decoding은 Dynamic Programming 기법의 일종이다. ( Viterbi Decoding에 관해 잘 정리된 글 : [https://ratsgo.github.io/data%20structure&algorithm/2017/11/14/viterbi/](https://ratsgo.github.io/data structure&algorithm/2017/11/14/viterbi/) 를 참고해도 좋을 것 같다 )

우리는 이제 특정 시점의 $$z$$가 아닌 전체 시점의 $$Z$$에 대해 알고 싶다. 

$$k^{*} = \underset{k}{argmax}\;p(z^k=1 \mid X) = \underset{k}{argmax}\;p(z^k=1, X) $$



이는 다음과 같이 구할 수 있다.

$$\begin{align*}
V_t^k &= max_{z_1,...,z_{t-1}}p(x_1,...,x_{t-1},z_1,...,z_{t-1},x_t,z_t^k=1)\\
&= max_{z_1,...,z_{t-1}}p(x_t,z_t^k = 1 \mid z_{t-1})p(x_1,...,x_{t-2},z_1,...,z_{t-2},x_{t-1},z_{t-1})\\
&= max_{z_{t-1}}p(x_t,z_t^k=1 \mid z_{t-1})\; max_{z_1,...,z_{t-2}}p(x_1,...,x_{t-2},z_1,...,z_{t-2},x_{t-1},x_{t-1})\\
&= max_{i \in z_{t-1}}p(x_t,z_t^k=1 \mid z_{t-1}^i=1)V_{t-1}^i \\
&= max_{i \in z_{t-1}}p(x_t \mid z_t^k=1) p(z_t^k=1 \mid z_{t-1}^i=1)V_{t-1}^i\\
&= p(x_t \mid z_t^k=1)\; max_{i \in z_{t-1}}p(z_t^k=1 \mid z_{t-1}^i=1)V_{t-1}^i \\
&= b_{k,idx(x_t)}\; max_{i \in z_{t-1}}a_{i,k}V_{t-1}^i
\end{align*}$$



위 과정을 T 시점 까지 계속 하면 된다

<br>

그림으로 이해를 해보자

<img src="https://i.imgur.com/WzPD3id.png" width="850" /> 

https://i.imgur.com/WzPD3id.png

위 그림은 수송 비용을 나타낸 그림이다. start에서 end 까지 갈 수 있는 다양한 경로가 있으며, 경로에 따라 들게 되는 비용은 다르다. 여기서 각 단계별 최소 비용이 들게 하는 최적의 경로를 구할 수 있을 것이다. 여기서 최적의 경로를 $$trace$$라고 한다.

<br>

다시 돌아와서, $$V_t^k$$를 구해보자. ( 여기에는 2개의 memoization table이 필요한데, 그 중 하나는 probability이고 하나는 trace이다.  )



### Algorithm

1 ) $$V_1^k$$를 initialize한다

$$V_1^k = b_{k,x_1}\pi_k$$

<br>

2 ) 시점 T까지 다음을 반복한다 

$$V_t^k = b_{k,idx(x_t)}max_{i \in z_{t-1}}a_{i,k}V_{t-1}^i$$

$$trace_t^k = \underset{i \in z_{t-1}}{argmax}\;a_{i,k}V_{t-1}^i$$

<br>

3) $$P(X,Z^{*}) = max_kV_T^k,z_T^{*} = \underset{k}{argmax}V_T^k,z_{t-1}^{*} = trace_{t}^{z_t^{*}}$$를 반환한다

위 방식대로 연산을 하면, underflow 문제가 발생할 수 있다. (0~1 사이의 값이 매우 여러번 곱해지게 되므로) 따라서, 우리는 연산을 할때 여기서 log transformation을 해서 계산을 한다.

<br>

## 7. Baum Welch Algorithm

앞에서 얘기했듯, HMM에서 주로 풀게 되는 문제는 **3) Learning Question**, 즉 X만 주어졌을 때 나머지 파라미터들을 estimate하는 것이다. 이를 푸는 알고리즘이 바로 **Baum Welch Algorithm**이다. 쉽게 표현하자면, 그냥 단지 HMM을 EM-Algorithm으로 푸는 것이라고 볼 수 있다.

<br>

**E-step**과 **M-step**에 대해 결론부터 이야기하자면, 다음과 같다.



### 1 ) E-step

$$q^{t+1}(z) = p(Z\mid X, \pi^t, a^t, b^t)$$

위 식에 따라 $$Z$$를 update한다.



### 2 ) M-step

(1) $$\pi$$

$$\pi^{t+1}_i = \frac{p(z_1^i=1 \mid X,\pi^t, a^t, b^t)}{\sum_{j=1}^{K}p(z_1^i=1 \mid X,\pi^t, a^t, b^t)}$$



(2) $$a$$

$$a^{t+1}_{i,j} = \frac{\sum_{t=2}^{T}p(z_{t-1}^i=1,z_t^j=1 \mid X,\pi^t, a^t, b^t)}{\sum_{t=2}^{T}p(z_{t-1}^i=1 \mid X,\pi^t, a^t, b^t)}$$



(3) $$b$$

$$b^{t+1}_{i,j} = \frac{\sum_{t=1}^{T}p(z_{t}^i=1 \mid X,\pi^t, a^t, b^t)\delta(idx(x_t)=j)}{\sum_{t=1}^{T}p(z_{t}^i=1 \mid X,\pi^t, a^t, b^t)}$$

