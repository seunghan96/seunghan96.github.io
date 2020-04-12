---
title: (paper) Weights Uncertainty in Neural Networks
categories: [DL,STAT]
tags: [Bayesian Inference, Deep Learning, Bayes by Backprop]
excerpt: Bayes by Backprop
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# [ (paper) Weights Uncertainty in Neural Networks ]



## Abstract

해당 논문에서는 ***Bayes by Backprop***이라는 새로운 알고리즘을 제안한다. 핵심은 간단하다. 이 알고리즘은 neural net의 **weight를 하나의 고정된 값이 아닌, 하나의 "확률 분포"로써 학습**을 시킨다는 점에서 기존의 neural net과 차이점을 가진다. 이는 MNIST classification 문제에서 dropout보다 나은 regularization 성능을 보임을 확인하였다. 또한, non-linear regression 문제에서는 뛰어난 일반화(generalization) 성능을 보였고, RL에서 사용되는 Exploration-Exploitation trade-off도 반영할 수 있다는 것을 확인하였다.

이 방식(Bayes by Backprop)이 왜 뛰어난 성능을 보이는지 논문 내용을 정리해보면서 알아보자.

<br>

<img src="https://labs.imaginea.com/stage/content/images/2019/05/bnn.png" width="550" /> 



## 1. Introduction

다들 알다시피, 기존의 neural net은  과적합(overfitting)에 취약하다. Bayes by Backprop(이하 BBB)는 neural net에 variational Bayesian Learning의 도입을 통해 **weight에도 uncertainty**를 부여함으로써 ( 확률분포로 만듬으로써 ) 이 문제를 해결한다. ( weight에 uncertainty가 부여가되면, 자연스럽게 결과(y값)를 구할 때도, 단 하나의 값 만을 출력하는 것이 아니라, **그 값에 대한 uncertainty도 구할 수 있다는 점**이 이 알고리즘의 아주 큰 장점이 아닐까 싶다. )



이 논문에서 제안한 **3개의 motivations**은 다음과 같다.

- 1 ) regularization via a compression cost on the weights
- 2 ) richer representations and predictions from cheap model averaging
- 3 ) exploration in simple RL problmes ( such as contextual bandits )



기존에도 이러한 과적합 문제를 해결하기 위한 다양한 방안들이 제시되었다. ( ex. dropout, early stopping, weight decay 등 ). 여기에서는, Bayesian Inference를 통해 보다 효율적인 regularization 방안을 제시한다.



모든 weight는 확률 분포를 가진다. 따라서, BBB는 하나의 network만을 가지는 것이 아니라, 여러개의 network (ensemble of networks)를 가지게 된다.  ( unbiased Monte Carlo estimates of the gradients를 사용함으로써! 뒤에서 보다 자세히 이야기하겠다 ) 각각의 network는 확률분포에서 샘플링된 weight들을 가진다. 하지만 다른 ensemble 기법들과 다르게, 훈련해야하는 parameter 수는 딱 2배만 필요하다 ( mean & variance ).

<br>

앞으로 얘기할 내용들은 다음과 같다.

section 2 ) notation & standard learning in NN

section 3 ) variational Bayesian learning for NN & contributions

section 4 ) application to contextual bandit problems

section 5 ) empirical results on (1) classification, (2) regression, (3) bandit problem

<br>

## 2. Point Estimates of Neural Networks

Neural Net에 대해 많이들 알고 있을 것이기 때문에 간단히 이야기하겠다.

Neural net을 probabilistic model의 관점에서 보면, 다음과 같이 나타낼 수 있다.

$$P(y \mid x,w)$$

( $$w$$ : set of parameters/weights )

<br>

Classification 문제의 경우, $$P(y \mid x,w)$$는 cateogircal distribution이 되겠고, minimize해야 하는 loss function은 cross-entropy나 softmax loss가 될 것이다.

Regression 문제의 경우, $$P(y \mid x,w)$$는 Gaussian distribution이 되겠고, minimize해야 하는 loss function은 MSE가 될 것이다.

<br>

여기서 $$w$$는 MLE나 MAP 등의 방법에 의해 구할 수 있을 것이다.

**MLE** 

$$w^{MLE} = \underset{w}{argmax}\;logP(D\mid w) = \underset{w}{argmax}\sum_{i}logP(y_i \mid x_i,w)$$



**MAP**

$$w^{MAP} = \underset{w}{argmax}\;logP(w\mid D) = \underset{w}{argmax}logP(D \mid w) + logP(w)$$

( 위 식에서 $$w$$의 prior로 Gaussian으로 잡으면, L2-regularization이 되고, Laplace로 잡으면 L1-regularization이 된다 )

<br>

## 3. Being Bayesian by Backpropagation

Bayesian Inference는 다음 식을 구하고자 한다. $$P(w \mid D)$$

input X를 넣었을 때 나올 것으로 예상되는 output Y는 다음과 같이 표현할 수 있다.

$$P(\hat{y} \mid \hat{x}) = E_{P(w \mid D)}[P(\hat{y} \mid \hat{x},w)]$$

<br>

Variational Inference에서, 최적의 파라미터 $$\theta$$를 찾기 위해 다음과 같이 KL-divergence를 최소화했던 것을 기억할 것이다.



$$\begin{align*}
\theta^{*} &= \underset{\theta}{argmin}\;KL[q(w \mid \theta) \mid \mid P(w\mid D)]\\
&= \underset{\theta}{argmin}\;\int q(w \mid \theta)log \frac{q(w\mid \theta)}{P(w)P(D \mid w)}dw\\
&= \underset{\theta}{argmin}\;KL[q(w \mid \theta) \mid \mid P(w)] - E_{q(w \mid \theta)}[logP(D\mid w)]
\end{align*}$$



여기서 loss(cost) function는 "variational free energy" (혹은 expected lower bound )라고 부른다.

우리는 이 loss를 다음과 같이 간단히 표기할 것이다.

$$F(D,\theta) = KL[q(w \mid \theta) \mid \mid P(w)] - E_{q(w \mid \theta)}[logP(D\mid w)]$$



위 loss function을 두 부분으로 나누어 생각해볼 수 있다.

(1) **data** dependent part : $$- E_{q(w \mid \theta)}[logP(D\mid w)]$$

(2) **prior** dependent part : $$KL[q(w \mid \theta) \mid \mid P(w)]$$

<br>

### 3-1. Unbiased Monte Carlo gradients



우리는 MC를 이용하여 $$F(D ,\theta)$$를 다음과 같이 나타낼 수 있다.

$$\begin{align*}
F(D,\theta) &= KL[q(w \mid \theta) \mid \mid P(w)] - E_{q(w \mid \theta)}[logP(D\mid w)]\\
&= E_{q(w \mid \theta)}[\frac{q(w\mid \theta))}{p(w)}] - E_{q(w \mid \theta)}[logP(D\mid w)]\\
&\approx \sum_{i=1}^{n}logq(w^{(i)} \mid \theta) - logP(w^{i}) - logP(D \mid w^{(i)})
\end{align*}$$

( 위 식에서 $$w^{(i)}$$ 는 $$q(w^{(i)} \mid \theta)$$ 에서 샘플링된 i번째 MC 샘플을 뜻한다 )

<br>

### 3-2. Gaussian Variational Posterior

------------------------------------------------- 참고 --------------------------------------------------

특정 조건 하에서, Expectation의 미분 값은 다음과 같이 미분값의 Expectation으로 나타낼 수 있다.

$$\frac{\partial}{\partial \theta}E_{q(w\mid \theta)}[f(w,\theta)] = E_{q(\epsilon)} [\frac{\partial f(w,\theta)}{\partial w} \frac{\partial w}{\partial \theta} + \frac{\partial f(w,\theta)}{\partial \theta}]$$

( 여기서 $$\epsilon$$은 $$q(\epsilon)$$하에서 뽑힌 r.v이고, $$w = t(\theta,\epsilon)$$ 이라고 하자. 추가로, $$q(\epsilon) d\epsilon = q(w \mid \theta)dw$$ 이다 )



**증명**

$$\begin{align*}
\frac{\partial}{\partial \theta}E_{q(w\mid \theta)}[f(w,\theta)] &= \frac{\partial}{\partial \theta} \int f(w,\theta)q(w \mid \theta)dw \\
&= \frac{\partial}{\partial \theta} \int f(w,\theta)q(\epsilon)d \epsilon \\
&= E_{q(\epsilon)} [\frac{\partial f(w,\theta)}{\partial w} \frac{\partial w}{\partial \theta} + \frac{\partial f(w,\theta)}{\partial \theta}]\\
\end{align*}$$



위 식을, 우리의 Loss Function $$\sum_{i=1}^{n}logq(w^{(i)} \mid \theta) - logP(w^{i}) - logP(D \mid w^{(i)})$$의 기울기를 구할 때 적용할 것이다.

 ( Let $$f(w,\theta) = logq(w \mid \theta) - logP(w)P(D \mid w)$$ )

------------------------------------------------------------------------------------------------------------

<br>

variational posterior가 diagonal Gaussian Distribution이라고 가정하자.

 $$\epsilon \sim N(0,I)$$ 

<br>

그리고 weight $$w$$를 평균이 $$\mu$$이고 표준 편차가 $$\sigma$$ 인 분포에서 sampling 된 것이라고 하자. 

우리는 $$\sigma$$가 non-negative하게 만들기 위해 다음과 같이 새로운 parameter $$\rho$$ 를 도입한다.

$$\sigma = log(1 + exp(\rho))$$

$$w = t(\theta, \epsilon) = \mu + log(1+exp(\rho))\circ \epsilon$$ 

 ( 여기서 $$\circ$$ 는 point-wise multiplication이다 )

<br>

**따라서, optimization 과정은 다음과 같이 정리될 수 있다.**

- 1 ) Sample $$\epsilon ~ N(0,I)$$

- 2 ) Let $$w = \mu + log(1+exp(\rho)) \circ \epsilon$$

- 3 ) Let $$\theta = (\mu, \rho)$$

- 4 ) Let $$f(w,\theta) = log\;q(w\mid \theta) - log\;P(w)P(D\mid w)$$

- 5) ( 위의 [참고]에 따라 .... )

  - 5-1) Calculate gradient w.r.t $$\mu$$

    $$\triangle_{\mu} = \frac{\partial f(w,\theta)}{\partial w} + \frac{\partial f(w,\theta)}{\partial \mu}$$ 
    

  - 5-2 )  Calculate gradient w.r.t $$\rho$$

    $$\triangle_{\rho} = \frac{\partial f(w,\theta)}{\partial w}\frac{\epsilon}{1+exp(-\rho)} + \frac{\partial f(w,\theta)}{\partial \rho}$$

- 6 ) Update parameters

  - $$\mu \leftarrow \mu - \alpha \triangle_{\mu}$$
  - $$\rho \leftarrow \rho - \alpha \triangle_{\rho}$$

<br>

### 3-3. Scale mixture prior

BBB는 prior로 하나의 Gaussian Distribution을 사용하지 않고, 다음과 같이 2개의 Gaussian Distribution mixture를 사용한다 ( 둘 다 mean은 0, but 서로 다른 variance )

$$P(w) = \prod_j \pi N(w_j \mid 0, \sigma_1^{2}) + (1-\pi)N(w_j \mid 0, \sigma_2^{2})$$

<br>

### 3-4. Mini-batches and KL re-weighting

우리의 학습 data $$D$$는 M개의 mini-batch로 나누어진 뒤, 각 배치 내에서 각각의 data의 gradient들이 평균내어진다.( 이 때 데이터를 꼭 동일한 개수의 M개의 batch로 나눌 필요는 없다. ) 여러 개의 mini-batch로 나눈 뒤의 loss는 다음과 같이 나타낼 수 있다.

$$F^{\pi}_{i}(D_i,\theta) = \pi_iKL[q(w \mid \theta) \mid \mid P(w)] - E_{q(w \mid \theta)}[logP(D_i\mid w)]$$

( $$E_M[\sum_{i=1}^{M}F_i^{\pi}(D_i,\theta)] = F(D,\theta)$$ 를 만족한다 )

<br>

이 논문에서는, $$\pi_i$$를 다음과 같이 설정할 경우 좋은 성능이 나온 다는 것을 확인했다.

$$\pi_i = \frac{2^{M-i}}{2^M-1}$$ 

<br>

이것은 무엇을 의미할까?

학습 과정에 있어서, 처음으로 들어가게 되는 mini batch들은 "complexity cost"에 의해 영향을 많이 받고, 점점 데이터가 많아 질수록 뒤에 들어가게 되는 mini batch들은"data"에 의해 영향을 많이 받는다는 것을 알 수 있다. ( 점차 prior의 영향을 덜 받고, data에 의해 영향을 더 많이 받게 된다 )

<br>

## 4. Contextual Bandits

( Bandits에 관해서 잘 정리되어 있는 블로그이다. 개념이 익숙치 않다면 읽어봐도 좋을 것 같다. https://daheekwon.github.io/mab1/ )

### 4-1. Thompson Sampling for Neural Networks

Thompson Sampling은 exploration과 exploitation 사이의 trade-off를 감안하여 sampling하는 기법이다. (강화학습에서 자주 사용된다) 간단히 설명하자면, 다음과 같은 algorithm을 가진다.

- 1 ) set of parameters에서 샘플링
- 2 ) sampling된 parameter들에 따라 가장 높은 expected reward를 가져다 주는 action을 선택
- 3 ) model을 update하고, 다시 1)로!



이것은 BBB에서도 사용된다.

- 1 ) weight를 sampling한다 ( $$w \sim q(w\mid \theta)$$ )
- 2 ) context $$x$$를 받는다
- 3 ) $$E_{P(r \mid x, a, w)}[r]$$ 를 최소화하는 행동 $$a$$를 선택한다
- 4 ) reward $$r$$를 받는다
- 5 ) variational parameter $$\theta$$를 update하고 다시 1)로 돌아간다



처음에는 uniform하게 행동을 하겠지만, 시간이 갈수록 agent의 행동은 가장 높은 보상을 주는 행동으로 점점 수렴할 것이다. 

<br>

## 5. Experiments

BBB는 다음과 같은 task에서 좋은 성능을 보이는 것으로 확인되었다.

- 1 ) MNIST classification
- 2 ) Non-linear regression 
- 3 ) Contextual Bandits