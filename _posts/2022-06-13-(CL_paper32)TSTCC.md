---
title: (paper 32) TS-TCC
categories: [CL, TS]
tags: []
excerpt: 2021
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Time-Series Representation Learning via Temporal and Contextual Contrasting

<br>

## Contents

0. Abstract
1. Introduction
2. Methods
   1. TS Data Augmentation
   2. Temporal Contrasting
   3. Contextual Contrasting

<br>

# 0. Abstract

propose an unsupervised **TS-TCC**

- **T**ime-**S**eries representation learning framework via **T**emporal and **C**ontextual **C**ontrasting

- (1) raw TS are transformed with **2 data augmentations**
  - weak augmentation
  - strong augmentation
- (2) novel **temporal contrasting module** 
  - to learn **robust** temporal representation
  - by designing **cross-view prediction**
- (3) propose **contextual contrasting module**
  - to learn **discriminative respresentations**

<br>

# 1. Introduction

Contrastive Learning

- mostly done in CV
- but not in TS

<br>

Why not in TS?

- (1) may not be able to address the **temporal dependencies** of data

- (2) some augmentation techniques **generally cannot fit well with TS**

<br>

propose TS-TCC

- employs **simple & effective data augmentations**
- propose novel **temporal contrastive module**
- propose **contextual contrasting module**

<br>

# 2. Methods

![figure2](/assets/img/cl/img62.png)

<br>

## (1) TS Data Augmentation

Using different augmentations can improve robustness of learned representations

<br>

Augmentation

- **weak : jitter-and-scale**
  - jitter = add random variations to the signal
  - scale = scale up its magnitude
- **strong : permutation-and-jitter**
  - permutation : split the signal into random \# of segments ( max \# = M ) & random shuffle
  - jitter = random jittering

<br>

Notation

- input sample : $$x$$
  - strongly augmented view : $$x^s \sim \mathcal{T}_s$$
  - weakly augmented view : $$x^w \sim \mathcal{T}_w$$
- encoder : $$\mathbf{z}=f_{\text {enc }}(\mathbf{x})$$
  - where $$\mathbf{z}=\left[z_1, z_2, \ldots z_T\right]$$
    - strongly augmented view : $$\mathbf{z}^s$$
    - weakly augmented view : $$\mathbf{z}^w$$

<br>

## (2) Temporal Contrasting

given latent variable $$\mathbf{z}$$,

use autoregressive model $$f_{a r}$$ to summarize all $$\mathbf{z}_{\leq t}$$ into a context vector $$c_t=f_{a r}(\mathbf{z} \leq t), c_t \in \mathbb{R}^h$$

- $$c_t$$ is used to predict the timesteps from $$z_{t+1}$$ until $$z_{t+k}(1<k \leq K)$$
- use **log-bilinear model**
  -  $$f_k\left(x_{t+k}, c_t\right)=\exp \left(\left(\mathcal{W}_k\left(c_t\right)\right)^T z_{t+k}\right)$$.

<br>

Cross-view prediction task

- use $$c_t^s$$ to predict future timesteps of weak augmentation $$z_{t+k}^w$$
- use $$c_t^w$$ to predict future timesteps of strong augmentation $$z_{t+k}^s$$

<br>

Contrastive Loss

- **minimize** dot product between **the predicted representation & true one of same sample**
- **maximize** dot product with **other samples $$\mathcal{N}_{t,k}$$**

$$\begin{aligned}
&\mathcal{L}_{T C}^s=-\frac{1}{K} \sum_{k=1}^K \log \frac{\exp \left(\left(\mathcal{W}_k\left(c_t^s\right)\right)^T z_{t+k}^w\right)}{\sum_{n \in \mathcal{N}_{t, k}} \exp \left(\left(\mathcal{W}_k\left(c_t^s\right)\right)^T z_n^w\right)} \\
&\mathcal{L}_{T C}^w=-\frac{1}{K} \sum_{k=1}^K \log \frac{\exp \left(\left(\mathcal{W}_k\left(c_t^w\right)\right)^T z_{t+k}^s\right)}{\sum_{n \in \mathcal{N}_{t, k}} \exp \left(\left(\mathcal{W}_k\left(c_t^w\right)\right)^T z_n^s\right)}
\end{aligned}$$.

<br>

Use transformer as the AR model

<br>

## (3) Contextual Contrasting

- to learn more **discriminative representations**

- $$2N$$ contexts
  - positive pair : $$\left(c_t^i, c_t^{i^{+}}\right)$$
  - negative pair : remaining $$(2 N-2)$$ pairs
- loss function :
  - $$\mathcal{L}_{C C}=-\sum_{i=1}^N \log \frac{\exp \left(\operatorname{sim}\left(c_t^i, c_t^{i^{+}}\right) / \tau\right)}{\sum_{m=1}^{2 N} \mathbb{1}_{[m \neq i]} \exp \left(\operatorname{sim}\left(c_t^i, c_t^m\right) / \tau\right)}$$,
    - where $$\operatorname{sim}(\boldsymbol{u}, \boldsymbol{v})=\boldsymbol{u}^T \boldsymbol{v} / \mid \mid \boldsymbol{u} \mid \mid  \mid \mid \boldsymbol{v} \mid \mid $$
- Overall self-supervised loss :
  - $$\mathcal{L}=\lambda_1 \cdot\left(\mathcal{L}_{T C}^s+\mathcal{L}_{T C}^w\right)+\lambda_2 \cdot \mathcal{L}_{C C}$$.

