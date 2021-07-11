---
title: \[reliable\] (paper 8) Can You Trust Your Model's Uncertainty\? Evaluating Predictive Uncertainty Under Data Shift
categories: [RELI,STUDY]
tags: [Reliable Learning]
excerpt: 
---

# Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Data Shift

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Contents

0. Abstract
1. Introduction
2. Background
   1. Notation
   2. Shift
   3. High-level overview of existing methods
3. Methods and Metrics
   1. 사용한 모델 종류들
   2. 사용한 metric 종류들

<br>

# 0. Abstract

Uncertainty를 측정하는 것은 매우 중요하다!

해당 prediction값을 믿을지 말지에 대해 결정적인 역할을 하기 때문!

<br>

# 1. Introduction

DNNs에서 predictive uncertainty를 quantify하기 위해 다양한 방법들이 제안 됨 ( 논문 pg2 참고 )

이 논문은, 데이터 분포의 변화 ( **distributional shift** ) condition 하에서의 **predictive uncertainty를 측정**하는 것을 제안한다!

<br>

### Contributions

provide a **benchmark for evaluating uncertainty** , not only on i.i.d setting, 

but also **uncertainty under DISTRIBUTIONAL SHIFT**

<BR>

# 2. Background

## 2-1) Notation

- $$x \in \mathbb{R}^{d}$$ : $$d$$ -dimensional features 
- $$y \in \{1, \ldots, k\}$$ : labels (targets) for $$k$$ -class classification
- assume **I.I.D**  samples, $$\mathcal{D}=\left\{\left(\boldsymbol{x}_{n}, y_{n}\right)\right\}_{n=1}^{N}$$

- $$p^{*}(x, y)$$ : unknown true distribution

  ( = **data generating process** )

<br>

Neural Network를 사용하여 $$p_{\boldsymbol{\theta}}(y \mid \boldsymbol{x})$$ 를 모델링한다!

- test time 때, (1) **training dataset과 같은 분포에서 sample한 test set**을 사용하여 evaluate
- 이 뿐만 아니라. (2) $$q(\boldsymbol{x}, y) \neq p^{*}(\boldsymbol{x}, y)$$ 에서 sample한 **O.O.D input에 대해서도** evaluate

<br>

## 2-2) Shift

2 종류의 shift를 고려한다.

- 1) $$k$$ 개의 class 중 하나로 shift 
  - ex) corruption, perturbation
  - 이러한 shift를 **covariate shift**라고도 부름
- 2) $$k$$ 개의 class가 아닌, 아예 새로운 class로 shift 
  - model이 이러한 새로운 instance에 대해서 **더 높은 predictive uncertainty**를 보이는지 확인한다.

<br>

## 2-3) High-level overview of existing methods

지금까지 uncertainty를 측정하거나, OOD detection을 하기 위한 다양한 방법론들이 나왔는데, 크게 아래와 같이 3가지로 구분 될 수 있다.

- 1) $$p(y \mid x)$$ 만을 사용하는 방법
- 2) joint distribution $$p(y, x)$$를 모델링하는 방법
- 3) OOD-detection을 하는 component를 $$p(y \mid x)$$에 추가한 방법

<br>

# 3. Methods and Metrics

## 3-1) 사용한 모델 종류들

- (Vanilla) Maximum softmax probability
- (Temp Scaling) Post-hoc calibration by temperature scaling using a validation set
- (Dropout) Monte-Carlo Dropout 
- (Ensembles) Ensembles of M networks trained independently on the entire dataset using random
  initialization 
- (SVI) Stochastic Variational Bayesian Inference for deep learning
- (LL) Approx. Bayesian inference for the parameters of the last layer only
  - (LL SVI) Mean field stochastic variational inference on the last layer only
  - (LL Dropout) Dropout only on the activations before the last layer

<br>

## 3-2) 사용한 metric 종류들

- classification accuracy
- NLL (Negative Log Likelihood)
- Brier Score
  - squared error of predicted probability vector $$p\left(y \mid x_{n}, \boldsymbol{\theta}\right)$$ & OH encoded true response $$y_n$$
  - $$\mathrm{BS}= \mid \mathcal{Y} \mid ^{-1} \sum_{y \in \mathcal{Y}}\left(p\left(y \mid \boldsymbol{x}_{n}, \boldsymbol{\theta}\right)-\delta\left(y-y_{n}\right)\right)^{2}= \mid \mathcal{Y} \mid ^{-1}\left(1-2 p\left(y_{n} \mid \boldsymbol{x}_{n}, \boldsymbol{\theta}\right)+\sum_{y \in \mathcal{Y}} p\left(y \mid \boldsymbol{x}_{n}, \boldsymbol{\theta}\right)^{2}\right)$$.
- Expected Calibration Error (ECE)
  - $$\mathrm{ECE}=\sum_{s=1}^{S} \frac{ \mid B_{s} \mid }{N} \mid \operatorname{acc}\left(B_{s}\right)-\operatorname{conf}\left(B_{s}\right) \mid $$.
    - $$\operatorname{acc}\left(B_{s}\right.) =  \mid B_{s} \mid ^{-1} \sum_{n \in B_{s}}\left[y_{n}=\hat{y}_{n}\right]$$.
    - $$\operatorname{conf}\left(B_{s}\right)= \mid B_{s} \mid ^{-1} \sum_{n \in B_{s}} p\left(\hat{y}_{n} \mid \boldsymbol{x}_{n}, \boldsymbol{\theta}\right)$$.
    - $$\hat{y}_{n}=\arg \max _{y} p(y \mid x)$$.

<br>

미완성



