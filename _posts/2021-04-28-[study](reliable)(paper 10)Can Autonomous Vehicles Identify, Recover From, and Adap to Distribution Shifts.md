---
title: \[reliable\] (paper 10) Can Autonomous Vehicles Identify, Recover From, and Adapt to Distribution Shifts\?
categories: [RELI,STUDY]
tags: [Reliable Learning]
excerpt: Augmix
---

# Can Autonomous Vehicles Identify, Recover From, and Adapt to Distribution Shifts?

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Contents

0. Abstract
1. Introduction
2. Problem Setting and Notation
3. Robust Imitative Planning (RIP)
   1. Bayesian Imitative Model
   2. Detecting Distribution Shifts
   3. Planning Under Epistemic Uncertainty
4. Benchmarking Robustness to Novelty

<br>

# 0. Abstract

key point: **자율주행 (AD)에서, detection & adaptation to O.O.D하기**

이 논문에서는 **RIP(Robust Imitative Planning)** 방법을 제안

- epistemic uncertainty-aware planning method

- detect & recover from distribution shifts
- reduce overconfident & catastrophic extrapolations in OOD sceneces

<br>

# 1. Introduction

domain : Autonomous driving (AD, 자율주행)

주요 문제 :

- ***reliability of ML models degrades radically, when exposed to NOVEL settings***
- 여기서 novel setting이란, 학습때는 보지 못했던 , OOD의 test 데이터

<br>

### Contribution

1. Epistemic uncertainty aware planning
   - RIP를 제안함 ( for detecting & recovering from distribution shifts )
   - deep ensemble을 사용하여 epistemic uncertainty에 대한 측정 가능
2. Uncertainty-driven online adaptation
   - online method인 AdaRIP (Adaptive RIP)를 제안
   - efficiently query the expert for feedback
   - real-world에 적용 가능
3. Autonomous car novel-scene benchmark

<br>

# 2. Problem Setting and Notation

"sequential" decision making 상황

4가지 가정

- [가정 1] Expert demonstration

  - dataset $$\mathcal{D}=\left\{\left(\mathrm{x}^{i}, \mathrm{y}^{i}\right)\right\}_{i=1}^{N}$$에 대한 access 가능

  - $$x$$ : 장면 (scence)

  - $$y$$ : time-profiled expert trajectories (i.e., plans)

    ( 이 trajectories 는 expert policy에서 sample된다 ... $$\mathbf{y} \sim \pi_{\text {expert }}(\cdot \mid \mathbf{x})$$ )

  - 목표 : unknown expert policy $$\pi_{\text {expert }}$$를 근사하기!

- [가정 2] Inverse Dynamics

- [가정 3] Global planner

  - global navigation system에 access 가능

- [가정 4] Perfect localization

<br>

# 3. Robust Imitative Planning (RIP)

아래의 3가지를 갖춘 imitation learning method를 추구한다

- 1) provides **distribution over expert plans**
- 2) quantifies **epistemic uncertainty** to allow **detection of OOD**
- 3) enables **robustness to distribution shift** with an explicit mechanism for recovery

<br>

## 3-1) Bayesian Imitative Model

- context-conditioned density estimation을 수행
- probabilistic imitative model $$q(\mathbf{y} \mid \mathbf{x} ; \boldsymbol{\theta})$$를 사용 ( MLE 통해 학습 )
  - $$\boldsymbol{\theta}_{\mathrm{MLE}}=\underset{\boldsymbol{\theta}}{\arg \max } \mathbb{E}_{(\mathrm{x}, \mathbf{y}) \sim \mathcal{D}}[\log q(\mathbf{y} \mid \mathbf{x} ; \boldsymbol{\theta})]$$.
- [ prior ] $$p(\boldsymbol{\theta})$$
  - induce distribution over density models $$q(\mathbf{y} \mid \mathbf{x} ; \boldsymbol{\theta})$$
- [ posterior ] $$p(\boldsymbol{\theta} \mid \mathcal{D})$$

<br>

### Practical Implementation

Autoregressive Neural Density Estimator (2018)를 사용한다

![figure2](/assets/img/reli/img16.png)

$$\begin{aligned}
q(\mathbf{y} \mid \mathbf{x} ; \boldsymbol{\theta}) &=\prod_{t=1}^{T} p\left(s_{t} \mid \mathbf{y}_{<t}, \mathbf{x} ; \boldsymbol{\theta}\right) \\
&=\prod_{t=1}^{T} \mathcal{N}\left(s_{t} ; \mu\left(\mathbf{y}_{<t}, \mathbf{x} ; \boldsymbol{\theta}\right), \Sigma\left(\mathbf{y}_{<t}, \mathbf{x} ; \boldsymbol{\theta}\right)\right)
\end{aligned}$$.

- $$\mu(\cdot ; \boldsymbol{\theta})$$ & $$\Sigma(\cdot ; \boldsymbol{\theta})$$ : RNN

- Normal은 unimodal하긴 하지만, autoregression 통해 Multi-modal 가능

  ( mixture of density networks, normalizing flow 등을 통해서도 가능 )

- exact inference는 intractable하다! 따라서 ensemble하여 approximation

<br>

## 3-2) Detecting Distribution Shifts

log likelihood of plan $$\log q(\mathbf{y} \mid \mathbf{x} ; \boldsymbol{\theta})$$ 통해서 paln의 quality 측정

구체적으로, 위의 Variance를 사용!

$$u(\mathbf{y}) \triangleq \operatorname{Var}_{p(\boldsymbol{\theta} \mid \mathcal{D})}[\log q(\mathbf{y} \mid \mathbf{x} ; \boldsymbol{\theta})]$$

- **disagreement of the qualities of a plan**, under model coming from the **posterior** $$p(\boldsymbol{\theta} \mid \mathcal{D})$$

- In-distribution의 경우 low variance

- OOD scene에서는 high variance

<br>

## 3-3) Planning Under Epistemic Uncertainty

planning to goal location $$\mathcal{G}$$ , under epistemic uncertainty ( = posterior $$p(\boldsymbol{\theta} \mid \mathcal{D})$$ )

![figure2](/assets/img/reli/img17.png)

- $$p(\boldsymbol{\theta} \mid \mathcal{D})$$ : uncertainty about TRUE EXPERT MODEL

<br>

### (a) Worst Case Model (RIP-WCM)

가장 최악의 상황의 model :

- $$s_{\mathrm{RIP}-\mathrm{WCM}} \triangleq \underset{\mathrm{y}}{\arg \max } \min _{\theta \in \operatorname{supp}(p(\theta \mid \mathcal{D}))} \log q(\mathbf{y} \mid \mathbf{x} ; \boldsymbol{\theta})$$.

<br>

### (b) Model Averaging (RIP-MA)

weighted average! 여기서 weight는 model's contribution , according to **posterior probability**

- $$s_{\mathrm{RIP}-\mathrm{MA}} \triangleq \underset{\mathbf{y}}{\arg \max } \int p(\theta \mid \mathcal{D}) \log q(\mathbf{y} \mid \mathbf{x} ; \boldsymbol{\theta}) \mathrm{d} \theta$$.

<br>

### 알고리즘 요약

![figure2](/assets/img/reli/img18.png)

<br>

# 4. Benchmarking Robustness to Novelty

생략