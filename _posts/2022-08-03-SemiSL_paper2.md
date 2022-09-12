---
title: (paper) SSL02 - Pseudo Label
categories: [ML]
tags: []
excerpt: 2013
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks (2013)

<br>

## Contents

0. Abstract
1. Introduction
2. Pseudo-Label Method for DNN
   1. Deep NN
   2. DAE (Denoising Auto-Encoder)
   3. Dropout
   4. Pseudo-Label
3. Why could Pseudo-Label work?
   1. Low-Density Separation between Classes
   2. Entropy Regularization
   3. Training with Pseudo-label as Entropy Regularization

<br>

# 0. Abstract

simple & efficient method of semi-supervised learning for DNN

- trained in supervised fashion, with both **labeled & unlabeled**

<br>

***Pseudo-Labels***

- labels of unlabeled data, chosen with maximum predicted probability
- **treat as if they were true labels**

<br>

# 1. Introduction

Pseudo-labels :

- favors a **low-density separation** between classes, a commonly assumed prior for semi-supervised learning. 

$$\rightarrow$$ same effect to Entropy Regulariaztion

<br>

**Entropy Regularization**

- commonly used prior for **semi-supervised learning**

- **conditional entropy** of the class prob : measure of **class overlap**

- **minimizing the entropy** for unlabeled data

  = **reducing the overlap of class pdf**

<br>

# 2. Pseudo-Label Method for DNN

## (1) Deep NN

- skip

<br>

## (2) DAE (Denoising Auto-Encoder)

[ Encoding ]

$$h_i=s\left(\sum_{j=1}^{d_v} W_{i j} \widetilde{x}_j+b_i\right)$$.

- $$\widetilde{x}_j$$ : corrupted version of $$j$$th input

<br>

[ Decoding ]

$$\widehat{x}_j=s\left(\sum_{i=1}^{d_h} W_{i j} h_i+a_j\right)$$.

<br>

## (3) Dropout

- skip

<br>

## (4) Pseudo-Label

Target classes for unlabeled data

$$y_i^{\prime}= \begin{cases}1 & \text { if } i=\operatorname{argmax}_{i^{\prime}} f_{i^{\prime}}(x) \\ 0 & \text { otherwise }\end{cases}$$.

<br>

use Pseudo-label in **fine-tuning** phase

- retrain pre-trained network with both **labeled & unlabeled** data

<br>

Total \# of labeled & unlabeled data is different

$$\rightarrow$$ ***Training balance is important***

<br>

Overall loss function

$$L=\frac{1}{n} \sum^n \sum_{i=1}^C L\left(y_i^m, f_i^m\right)+\alpha(t) \frac{1}{n^{\prime}} \sum^{n^{\prime}} \sum_{i=1}^C L\left(y_i^{\prime m}, f_i^{\prime m}\right)$$.

- $$n$$ : number of labeled data
- $$n^{‘}$$ : number of unlabeled data

- $$\alpha(t)$$ : coefficient of balance between the two
  - high $$\rightarrow$$ disturbs labeled data
  - small $$\rightarrow$$ no benefit of pseudo-labeling

<br>

Settings

$$\alpha(t)= \begin{cases}0 & t<T_1 \\ \frac{t-T_1}{T_2-T_1} \alpha_f & T_1 \leq t<T_2 \\ \alpha_f & T_2 \leq t\end{cases}$$.

- with $$\alpha_f=3, T_1=100, T_2=600$$ without pre-training
-  $$T_1=200, T_2=800$$ with DAE.

<br>

# 3. Why could Pseudo-Label work?

## (1) Low-Density Separation between Classes

***cluster assumption***

-  the **decision boundary** should lie in **low-density regions** to improve generalization performance

<br>

## (2) Entropy Regularization

means to benefit from unlabeled data, in the framework of **MAP estimation**

- minimizing the conditional entropy of class probabilities of unlabeled data

<br>

MAP estimate :

- $$C(\theta, \lambda)=\sum_{m=1}^n \log P\left(y^m \mid x^m ; \theta\right)-\lambda H\left(y \mid x^{\prime} ; \theta\right)$$
  - where $$H\left(y \mid x^{\prime}\right)=-\frac{1}{n^{\prime}} \sum_{m=1}^{n^{\prime}} \sum_{i=1}^C P\left(y_i^m=1 \mid x^{\prime m}\right) \log P\left(y_i^m=1 \mid x^{\prime m}\right)$$

<br>

## (3) Training with Pseudo-label as Entropy Regularization

Pseudo Label = Entropy Regularization

- pseudo label : encourages the predicted class probabilities to be near 1-of-K code

<br>

[Entropy Regularzation]

$$C(\theta, \lambda)=\sum_{m=1}^n \log P\left(y^m \mid x^m ; \theta\right)-\lambda H\left(y \mid x^{\prime} ; \theta\right)$$.

<br>

[Pseudo Label]

$$L=\frac{1}{n} \sum^n \sum_{i=1}^C L\left(y_i^m, f_i^m\right)+\alpha(t) \frac{1}{n^{\prime}} \sum^{n^{\prime}} \sum_{i=1}^C L\left(y_i^{\prime m}, f_i^{\prime m}\right)$$

<br>

Equivalence

- $$\sum_{m=1}^n \log P\left(y^m \mid x^m ; \theta\right)$$ & $$\frac{1}{n} \sum^n \sum_{i=1}^C L\left(y_i^m, f_i^m\right)$$
- $$-\lambda H\left(y \mid x^{\prime} ; \theta\right)$$ & $$\alpha(t) \frac{1}{n^{\prime}} \sum^{n^{\prime}} \sum_{i=1}^C L\left(y_i^{\prime m}, f_i^{\prime m}\right)$$

