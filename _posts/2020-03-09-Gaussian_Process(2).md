---
title: Gaussian Process (2) GP Implementation
categories: [STAT]
tags: [Bayesian,Statistics,GP]
excerpt: GP(2) - GP Implementation
---

# Gaussian Process (2)

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( https://github.com/aailabkaist/Introduction-to-Artificial-Intelligence-Machine-Learning/blob/master/를 참고하여 공부하고 정리하였습니다 )

## 4. Random Process

### (1) What is Gaussian Process?

앞에서 Gaussian Process Regression에 대해서 알아보았다. 이제는 Gaussian Process에 대해서 이야기 할 것이다. Gaussian Process는 Random Process의 한 형태로, 우선 Random Process에 대해서 얘기해볼 것이다.

Random Process는 **index parameter (t) 에 대해, 다음과 같은 random variable의 collection의 process**라고 생각하면 된다. $$\{X(t) \mid t \in T\}$$

( 여기서 t는 주로 시간을 의미하지만, 꼭 시간이 아닌 일종의 연속적인 index로 생각해도 된다. )



사용하는 함수는 **다음과 같이 $$X(t,\omega)$$로 표현**할 수 있다

- $$\omega$$는 random experiment를 통해 나온 값( y값 )이라고 생각하면 된다. ( $$\omega \in \Omega$$ )

- 1 )  $$t$$를 고정시키면, $$X(t,\omega)$$는 $$\Omega$$에 대한 random variable이라고 볼 수 있다.
- 2 ) $$\omega$$를 고정시키면, $$X(t,\omega)$$ 는 $$t$$에 대한 sample function이라고 볼 수 있다.

<br>

Gaussian Process는 Random process의 하나의 종류로, 위의 $$X(t,\omega)$$에서 $$X$$가 Normal Distribution을 따르는 경우를 이야기한다.

우리가 이전 포스트에서 배운 식을 보면 이해할 수 있을 것이다.

$$P(T) = N(T \mid 0, (\beta I_N)^{-1} + K)$$

여기서 $$K_{nm} = k(x_n,x_m)$$이다



**example** )

$$K_{nm} = k(x_n,x_m)  = \theta_0 exp(-\frac{\theta_1}{2}\mid \mid x_n - x_m \mid \mid^2) + \theta_2 + \theta_3 x_n^Tx_m$$

<br>

### (2) Hyper-parameters of Gaussian Process Regression

우리의 Kernel Function이 다음과 같다고 해보자.

$$K_{nm} = k(x_n,x_m)  = \theta_0 exp(-\frac{\theta_1}{2}\mid \mid x_n - x_m \mid \mid^2) + \theta_2 + \theta_3 x_n^Tx_m$$

<br>

여기서 우리의 kernel hyperparameter $$( \theta_0, \theta_1, \theta_2, \theta_3, \beta)$$ 를 어떻게 잡냐에 따라 결과 값은 천차만별이다. 이 hyperparameter를 학습하지 않고 사용하면 좋은 성능이 나올리가 없다. 어떻게 최적의 hyperparameter를 찾을 수 있을까?



**Finding the best hyperparameter**

우선, $$P(T)$$를 다음과 같이 나타내보자.

$$P(T) = P(T \mid \theta) = N(T \mid 0, (\beta I_N)^{-1} +K) = N(T \mid 0,C)$$



우리는 이를 mle를 사용하여 풀 수 있다.

$$\frac{\partial}{\partial \theta_i} P(T \mid \theta) =0$$ 를 만드는 $$\theta$$ 찾기!



이는, 뒤에서 tensorflow를 통해서 solution을 구할 것이다.

<br>

### (3) GP with Tensorflow

코드로 구현하기전에, 우리의 kernel function $$K_{nm}$$과 $$P(T)$$를 정리해보자.





