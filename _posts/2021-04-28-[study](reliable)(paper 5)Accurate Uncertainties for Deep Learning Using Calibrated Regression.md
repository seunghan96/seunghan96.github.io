---
title: \[reliable\] (paper 5) Accurate Uncertainties for Deep Learning Using Calibrated Regression
categories: [RELI,STUDY]
tags: [Reliable Learning]
excerpt: Deep Learning Uncertainty, Deep Ensembles, Predictive Uncertainty
---

# Accurate Uncertainties for Deep Learning Using Calibrated Regression

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Contents

0. Abstract
1. Introduction
2. Calibrated Classification
   1. Calibration
   2. Training Calibrated Classifiers
3. Calibrated Regression
   1. Calibration
   2. Training Calibrated Regression Models
   3. Recalibrating Bayesian models
   4. Features for Recalibration
   5. Diagnostic Tools

<br>

# 0. Abstract

Goal : **Uncertainty 설명하기!**

- ex) Bayesian method
- 하지만, Bayesian method의 문제점 : **approximate** inference를 하기 때문에, estimate가 inaccurate

<br>

이 Paper는 그 어떠한 **regression algorithm을 simple하게 calibrate**하는 방법을 제안한다.

<br>

# 1. Introduction

이 논문은 **uncertainty estimation over CONTINOUS variable**에 대해 제안

<br>

기존의 Bayesian 방법의 문제점 : (아래 그림 참조)

![figure2](/assets/img/reli/img10.png)

- (상) [ 베이지안 ] interval fails to capture true distribution
- (하) [ 제안된 방법 ] recalibrated method
  - 90% C.I가 정확히 9/10의 point를 cover한다.

<br>
한 줄 요약 :

***propose a new procedure for RECALIBRATING any REGRESSION algorithm, that is inspired by Platt Scaling for classification***

<br>

### Contributions

- 1) simple technique for RE-calibrating the output of REGRESSION output

  ( classification을 위한 Platt scaling의 확장판 )

- 2) 이 technique을 BNN의 문제점을 푸는데에 사용함

  ( BNN = Bayesian Neural Network )

<br>

# 2. Calibrated Classification

Notation

- labeled dataset $$x_{t}, y_{t} \in \mathcal{X} \times \mathcal{Y}$$ for $$t=1,2, \ldots, T$$ 

- $$X, Y \sim \mathbb{P}$$, where $$\mathbb{P}$$ is data distribution

- forecaster $$H: \mathcal{X} \rightarrow(\mathcal{Y} \rightarrow[0,1])$$ 

  - outputs a probability distribution $$F_{t}(y)$$ targeting the label $$y_{t}$$

    ( $$Y$$ 가 continuous하면 $$F_{t}$$ 는 CDF )

<br>

## 2-1) Calibration

ex) binary classification

- $$\mathcal{Y}=\{0,1\}$$인 경우
- $$H$$ is calibrated  $$\leftrightarrow$$  $$\frac{\sum_{t=1}^{T} y_{t} \mathbb{I}\left\{H\left(x_{t}\right)=p\right\}}{\sum_{t=1}^{T} \mathbb{I}\left\{H\left(x_{t}\right)=p\right\}} \rightarrow p \text { for all } p \in[0,1]$$.

<br>

Calibration을 위한 충분 조건

- $$\mathbb{P}(Y=1 \mid H(X)=p)=p \text { for all } p \in[0,1]$$.

<br>

Calibration & Sharpness

- 둘 다 중요하다!
- sharp하다 = **probabilities should be close to 0 or 1**

<br>

## 2-2) Training Calibrated Classifiers

### (1) Estimating a probability distribution

Calibrated Classifier : $$R \circ H$$

- $$R(p)=\mathbb{P}(Y=1 \mid H(X)=p)$$ 잘 만들기

ex) **Platt scaling** 

- approximate $$R(p)=\mathbb{P}(Y=1 \mid H(X)=p)$$ with sigmoid

<br>

### (2) Projections & Features

Base classifier $$H$$ : $$H: \mathcal{X} \rightarrow \Phi$$

-  output features $$\phi \in \Phi \subseteq \mathbb{R}^{d}$$ that **do not correspond to probabilities**
- $$R: \Phi \rightarrow[0,1]$$ 를 $$\mathbb{P}(Y=1 \mid H(X)=\phi)$$에 적용

<br>

### (3) Diagnostic Tools

- calibration curve 사용하기 ( 아래 그림 참조 )
- group $$p_t$$ into intervals $$I_j$$ ( for $$j=1,...,m$$ ) , which are partitions of [0,1]
- calibration curve plots the predictive average $$p_j$$ in each interval $$I_j$$
  - $$p_{j}=T_{j}^{-1} \sum_{t: p_{t} \in I_{j}} p_{t}$$.

<br>

# 3. Calibrated Regression

Regression에서는,

- forecaster $$H$$ outputs at each step $$t$$ a CDF $$F_t$$, targeting $$y_t$$
- quantile function : $$F_{t}^{-1}(p)=\inf \left\{y: p \leq F_{t}(y)\right\}$$
  - $$F_{t}^{-1}:[0,1] \rightarrow \mathcal{Y}$$ .

<br>

## 3-1) Calibration

regression에서 calibration의 의미

- 90%의 횟수로, $$y_t$$는90% C.I에 위치해야
- $$\frac{\sum_{t=1}^{T} \mathbb{I}\left\{y_{t} \leq F_{t}^{-1}(p)\right\}}{T} \rightarrow p \text { for all } p \in[0,1]$$.

<br>

충분조건 : $$\mathbb{P}\left(Y \leq F_{X}^{-1}(p)\right)=p \text { for all } p \in[0,1]$$

- forecaster : $$F_{X}=H(X)$$

<br>

다른 표현으로 나타내면, 아래와 같다.

- $$\frac{\sum_{t=1}^{T} \mathbb{I}\left\{F_{t}^{-1}\left(p_{1}\right) \leq y_{t} \leq F_{t}^{-1}\left(p_{2}\right)\right\}}{T} \rightarrow p_{2}-p_{1}$$>

<br>

## 3-2) Training Calibrated Regression Models

**simple re-calibrated scheme**을 제안한다!

- pre-trained forecaster : $$H$$
- auxiliary model : $$R$$ : $$[0,1] \rightarrow [0,1]$$
- **CALIBRATED model** : $$R \circ F_{t}$$

<br>

### Algorithm

![figure2](/assets/img/reli/img11.png)

<br>

Estimating a probability distribution

- perfectly calibrated forecaster :

  $$R \circ F_{t}$$, where $$R(p):=\mathbb{P}(Y \leq \left.F_{X}^{-1}(p)\right)$$.

- 위의 cdf를 estimate하도록 formulate하기

<br>

### Example

$$p=95 \%$$, but only $$80 / 100$$ observed $$y_{t}$$ fall below the $$95 \%$$ quantile of $$F_{t}$$

$$\rightarrow$$ Adjust the $$95 \%$$ quantile to $$80 \%$$ 

![figure2](/assets/img/reli/img12.png)

- learn $$\mathbb{P}\left(Y \leq F_{X}^{-1}(p)\right)$$ by fitting any regression algorithm ( isotonic regression 추천 )
- 학습에 사용할 data : $$\left\{F_{t}\left(y_{t}\right), \hat{P}\left(F_{t}\left(y_{t}\right)\right)\right\}_{t=1}^{T}$$ 
- $$\hat{P}(p)=\frac{\left|\left\{y_{t} \mid F_{t}\left(y_{t}\right) \leq p, t=1, \ldots, T\right\}\right|}{T}$$.

<br>

## 3-3) Recalibrating Bayesian models

Probabilistic forecasts $$F_{t}$$ : BNN, GP등을 통해 찾음

- model : $$\mathcal{N}\left(\mu\left(x_{t}\right), \sigma^{2}\left(x_{t}\right)\right)$$.

- ex) MCDO 

그러나, true data distribution $$\mathbb{P}(Y \mid X)$$ 이 Gaussian이 아닐 경우...

$$\rightarrow$$ uncertainty estimates는 well calibrated되지 않을 것

<br>

## 3-4) Features for Recalibration

[Algorithm 1]을 사용하여 recalibration을 진행할 수 있음.

이를 아무런 increasing function $$F(y): \mathcal{Y} \rightarrow \Phi$$ where $$\Phi \subseteq \mathbb{R}$$ defines a "feature"  that correlates with the confidence of the classifier로 generalize 할 수 있음.

ex) distance from mean prediction ( $$\phi \in \Phi$$ )

- cdf : $$\mathbb{P}\left(Y \leq F_{X}^{-1}(\phi)\right)$$

- $$[H(x)](y)=F_{x}(y)=y-\mu(x)$$.

ex 2) uncertainty까지 고려하여..

- $$F_{x}(y)=(y-\mu(x)) / \sigma(x)$$.

<br>

## 3-5) Diagnostic Tools

### (a) Calibration

$$\operatorname{cal}\left(F_{1}, y_{1}, \ldots, F_{T}, y_{T}\right)=\sum_{j=1}^{m} w_{j} \cdot\left(p_{j}-\hat{p}_{j}\right)^{2}$$.

- $$\hat{p}_{j}=\frac{\left|\left\{y_{t} \mid F_{t}\left(y_{t}\right) \leq p_{j}, t=1, \ldots, T\right\}\right|}{T}$$.

<br>

### (b) Sharpness

- $$\operatorname{sha}\left(F_{1}, \ldots, F_{T}\right)=\frac{1}{T} \sum_{t=1}^{T} \operatorname{var}\left(F_{t}\right)$$.



