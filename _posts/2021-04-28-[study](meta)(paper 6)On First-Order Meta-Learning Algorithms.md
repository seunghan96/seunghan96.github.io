---
title: \[meta\] (paper 6) On First Order Meta Learning Algorithms
categories: [META,STUDY]
tags: [Meta Learning]
excerpt: MAML, FOMAML, Reptile
---

# On First-Order Meta-Learning Algorithms

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Contents

0. Abstract
1. Introduction
2. Meta-Learning an Initialization
3. Reptile

<br>

# 0. Abstract

Meta Learning

- distribution of **tasks**
- learn quickly ( 적은 unseen 데이터로도 )

<br>

이 논문은 new task에 빠르게 fine-tune 될 수 있는 family of algorithms 소개

- using only **FIRST ORDER DERIVATIVES** for the meta learning updates
- **FOMAML (First-Order MAML)**
  - approximation to MAML ( 2nd derivative 무시 )
- **Reptile**
  - 이 논문에서 제안한 new algorithm
  - repeatedly sample task & train & move weights .....

<br>

# 1. Introduction

적은 수의 데이터로도 학습하는데에 도움을 줄 수 있는 Meta Learning이 떠오름!

- Bayesian Inference $$\rightarrow$$ computationally intractable
- Meta Learning $$\rightarrow$$ DIRECTLY optimize a fast-learning algorithm

<br>

Meta Learning의 다양한 approaches

- 방법 1) learning algorithm이 RNN의 weight에 encode

- **방법 2) initialization & fine-tune**

  - 기존) large 데이터로 pre-train & small 데이터로 fine-tune

    ( 하지만, fine-tuning에 좋다는 guarantee X )

  - ex) MAML, FOMAML, Reptile

<br>

[ MAML, FOMAML, Reptile ]

- **MAML : directly optimizes performance w.r.t this initialization**

  - by differentiating through the fine-tuning process

    따라서, test time에 large number of gradient step 필요할때 좋지 않음

- **FOMAML (first-order MAML) : 2nd derivative term 무시한 MAML**

  - MiniImageNet 데이터셋에 MAML 못지 않은 성능

- **Reptile**

  - 이 논문에서 제안한 방법
  - FOMAML과 마찬가지로, ***based on first-order gradient***

<br>

**[ Contribution ]**

- 1) first-order MAML(FOMAML)은 생각보다 easy implementation
- 2) Reptile 소개
  - closely related to FOMAML ( 마찬가지로 simple 하다 )
    - 유사점 ) joint training ( = train to minimize loss on the expectation over training tasks )
    - 차이점 ) train-test split 불필요
- 3) FOMAML & Reptile의 theoretical analysis

<br>

# 2. Meta-Learning an Initialization

Optimization problem of MAML

- 1) initial set of parameters $$\phi$$

- 2) $$k$$번 update하고 나면, $$L_\tau$$가 낮아지도록

  - $$L_\tau$$ : 랜덤하게 샘플된 task $$\tau$$에 해당하는 loss

  ( 즉, $$\underset{\phi}{\operatorname{minimize}} \mathbb{E}_{\tau}\left[L_{\tau}\left(U_{\tau}^{k}(\phi)\right)\right]$$ )

<br>

$$\underset{\phi}{\operatorname{minimize}} \mathbb{E}_{\tau}\left[L_{\tau}\left(U_{\tau}^{k}(\phi)\right)\right]$$.

- $$U_{\tau}^{k}$$ : $$\phi$$를 $$k$$번 update 시키는 operator

  ( few-shot learning에서 gradient descent를 하는 역할 )

- MAML이 위를 해결하는 방법 :

  - inner loop optimization은 training sample $$A$$를 사용
  - loss 계산은 test sample $$B$$ 로
  - 즉, $$\underset{\phi}{\operatorname{minimize}} \mathbb{E}_{\tau}\left[L_{\tau, B}\left(U_{\tau, A}(\phi)\right)\right]$$

- MAML은 이를 SGD를 사용하여 optimize한다

  $$\begin{aligned}
  g_{\text {MAML }} &=\frac{\partial}{\partial \phi} L_{\tau, B}\left(U_{\tau, A}(\phi)\right) \\
  &=U_{\tau, A}^{\prime}(\phi) L_{\tau, B}^{\prime}(\widetilde{\phi}), \quad \text { where } \quad \widetilde{\phi}=U_{\tau, A}(\phi)
  \end{aligned}$$.

  - $$U_{\tau, A}^{\prime}(\phi)$$ : Jacobian matrix of $$U_{\tau, A}$$
    - $$U_{\tau, A}(\phi)=\phi+g_{1}+g_{2}+\cdots+g_{k}$$.
    - FOMAML는 이를 상수 취급한다 ( = $$U_{\tau, A}^{\prime}(\phi)$$를 identity matrix로 )

<br>

### FOMAML

MAML vs FOMAL

- $$g_{\text {MAML }}=U_{\tau, A}^{\prime}(\phi) L_{\tau, B}^{\prime}(\widetilde{\phi}), \quad \text { where } \quad \widetilde{\phi}=U_{\tau, A}(\phi)$$.
- $$g_{\text {FOMAML }}=L_{\tau, B}^{\prime}(\widetilde{\phi})$$.

<br>

Algorithm

- step 1) sample task  $$\tau$$
- step 2) apply the update operator, yielding $$\widetilde{\phi}=U_{\tau, A}(\phi)$$
- step 3) compute the gradient at $$\Phi, g_{\text {FOMAML }}=L_{\tau, B}^{\prime}(\tilde{\phi})$$
- step 4) plug $$g_{\text {FOMAML }}$$ into the outer-loop optimizer.

<br>

# 3. Reptile

간단 소개

- 새로운 **first-order** gradient-based meta-learning algorithm
- MAML과 마찬가지로, **learn a initialization**

![figure2](/assets/img/META/img20.png)