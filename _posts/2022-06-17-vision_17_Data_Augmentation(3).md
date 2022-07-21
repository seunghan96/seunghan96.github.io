---
title: 17. Data Augmentation (3) - AutoML based
categories: [CV]
tags: []
excerpt: AutoML based Data Augmentation
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( 참고 : 패스트 캠퍼스 , 한번에 끝내는 컴퓨터비전 초격차 패키지 )

# Data Augmentation (3)

Categories of Data Augmentation

- (1) **Rule-based**
- (2) **GAN-based**
- (3) **AutoML-based**

<br>

## (3) AutoML-based

1. AutoAugment
2. Population Based AutoAugment
3. Fast AutoAugment
4. Faster AutoAugment
5. RandAugment
6. UniformAugment
7. TrivialAugment

<br>

### 1) AutoAugment

( *AutoAugment : Learning Augmentation Policies from Data, Cubuk et al., CVPR 2019* )

- like **NAS (Neural Architecture Search)**, sample **Augmentation Policy** from RNN controller
- Reinforcment Learning / Policy gradient
  - reward : **validation accuracy**

![figure2](/assets/img/cv/cv251.png)

![figure2](/assets/img/cv/cv252.png)

![figure2](/assets/img/cv/cv250.png)

<br>

Cons : **TOO MUCH COMPUTATION TIME**

( $$\because$$ policy gradient based on **validation error** )

<br>

### 2) Population Based AutoAugment

( *Population Based Augmentation : Efficient Learning of Augmentation Policy Schedules, Ho et al., ICML 2019* )

Problem of **AutoAugment** : **computationally infeasible**

$$\rightarrow$$ solution : ***Population Based AutoAugment***

![figure2](/assets/img/cv/cv253.png)

<br>

Characteristics

- **non-stationary** augmentation policy schedules

  ( instead of fixed augmentaiton policy )

- Exploration & exploitation

$$\rightarrow$$ **outputs an augmentation policy!**

![figure2](/assets/img/cv/cv254.png)

![figure2](/assets/img/cv/cv255.png)

<br>

### 3) Fast AutoAugment

( *Fast AutoAugment, Lim et al., NeurIPS 2019* )

Problem of **AutoAugment** : **computationally infeasible**

![figure2](/assets/img/cv/cv256.png)

<br>

Solution : efficient search strategy, using **density matching**

- concept of **Bayesian Optimization** ( Tree-structued Parzen Estimator (TPE) )

![figure2](/assets/img/cv/cv257.png)

<br>

### 4) Faster AutoAugment

( *Faster AutoAugment : Learning Augmentation Strategies using Backpropagation, Hataya et al., ECCV 2020* )

Motivation

- make it **differentiable** ! **DIFFERENTIABLE AutoAugment**

![figure2](/assets/img/cv/cv258.png)

<br>

Candidates of operations :

![figure2](/assets/img/cv/cv259.png)

<br>

![figure2](/assets/img/cv/cv260.png)

![figure2](/assets/img/cv/cv261.png)

<br>

### 5) RandAugment

( *RandAugment : Practical automated data augmentation with a reduced search space, Cebuk et al., NeurIPS 2020* )

Why need **AutoML**? Too **LARGE search space!**

$$\rightarrow$$ Instead of searching, **random sample**

hyperparameter :

- $$N$$ : number of operations
- $$M$$ : range of operations

![figure2](/assets/img/cv/cv262.png)

![figure2](/assets/img/cv/cv263.png)

![figure2](/assets/img/cv/cv264.png)

<br>

### 6) UniformAugment

( *UniformAugment : A Search-free Probabilistic Data Augmentation Approach, LingChen et al.* )

- mix 2 images pixel wise ( NO SEARCH )
- train : **N-class multi-label prediction**
- Prob 0~1 of,,,
  - using certain augmentation (O,X)
  - Magnitude of augmentation

![figure2](/assets/img/cv/cv265.png)

<br>

### 7) TrivialAugment

( *Trivial Augment : Tuning-free Yet State-of-the-Art Data Augmentation, mUiller et al., ICCV 2021* )

previous methods

- consider trade-off between **efficiency & effectiveness**

<br>

Proposes…

- instead of parameter-free, just **search important factors**
  - (1) Augmentation Type
  - (2) Magnitude

![figure2](/assets/img/cv/cv266.png)

![figure2](/assets/img/cv/cv267.png)