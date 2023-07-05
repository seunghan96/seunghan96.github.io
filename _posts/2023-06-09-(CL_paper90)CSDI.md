---
title: (paper 90) CSDI; Conditional Score-baesd Diffusion Models for Probabilistic TS Imputation
categories: [TS]
tags: []
excerpt: 2021
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# CSDI: Conditional Score-baesd Diffusion Models for Probabilistic TS Imputation

<br>

## Contents

0. Abstract
1. Introduction
   


<br>

# 0. Abstract

**AR models** : natural for TS imputation

**Score-based diffusion models **: outperformed ARs, in...

- image generation and audio synthesis
- would be promising for TS imputation. 

<br>

### Conditional Score-based Diffusion models for Imputation (CSDI)

Novel **TS imputation** method 

- utilizes **score-based diffusion models** conditioned on **observed data** 
- conditional diffusion model, explicitly trained for **imputation**
- exploit correlations between observed values. 

<br>

# 1. Introduction

Imputation methods based on DNN

- (1) Deterministic imputation
- (2) Probabilistic imputation 

$\rightarrow$ typically utilize AR models to deal with TS

<br>

### Score-based diffusion models 

can also be used to impute missing values 

- by approximating the **scores of the posterior distribution**,

  obtained from the **prior by conditioning on the observed values**

- may work well in practice, they do not correspond to the exact conditional distribution.

<br>

### CSDI

- a novel probabilistic imputation method
- directly learns the conditional distn with conditional score-based diffusion models. 
- designed for imputation &  can exploit useful information in observed values. 

![figure2](/assets/img/cv/img429.png)

<br>

Overview

- Start imputation from random noise

- Gradually convert the noise into plausible TS
  - via reverse process $p_\theta$ of the conditional diffusion model. 

<br>

Reverse Process (at step $t$)

- removes noise from the output of the previous step $(t+1)$. 
- can take **observations as a conditional input**
  - exploit information in the observations for denoising. 
- utilize an attention mechanism 
  - to capture the temporal and feature dependencies of TS

<br>

Data

- observed values (i.e., conditional information)
- ground-truth missing values (i.e., imputation targets). 

<br>

However, in practice **we do not know the ground-truth missing values**

$\rightarrow$ inspired by MLM, develop a SSL method that separates observed values into conditional information and imputation targets. 

<br>

CSDI is formulated for general imputation tasks

( not restricted to TS imputation )

<br>

### Contributions

1. Conditional score-based diffusion models for probabilistic imputation (CSDI
   - to train the conditional diffusion model, develop SSL method
2. Experiments
   - improves the continuous ranked probability score (CRPS)
   - decreases the mean absolute error (MAE) by 5-20%
3. Can be applied to TS interpolations and probabilistic forecasting

<br>

# 2. Related works

## (1) TS imputation with DL

- RNN-based

- RNN + GANs/Self-training
- Combination of RNNs & attention mechanisms : successful

$\rightarrow$ mostly focused on deterministic imputation

( $\leftrightarrow$ GP-VAE : has been recently developed as a probabilistic imputation method )

<br>

## (2) Score-based Generative models

Examples)

- score matching with Langevin dynamics 

- denoising diffusion probabilistic models

outperformed existing methods with other deep generative models

<br>

TimeGrad 

- utilized diffusion probabilistic models for probabilistic TS forecasting.
- BUT cannot be applied to TS imputation
  -  due to the use of RNNs to handle past time series.

<br>

# 3. Background

## (1) MTS imputation

https://arxiv.org/pdf/2107.03502.pdf

We consider $N$ multivariate time series with missing values. Let us denote the values of each time series as $\mathbf{X}=\left\{x_{1: K, 1: L}\right\} \in \mathbb{R}^{K \times L}$ where $K$ is the number of features and $L$ is the length of time series. While the length $L$ can be different for each time series, we treat the length of all time series as the same for simplicity, unless otherwise stated. We also denote an observation mask as $\mathbf{M}=\left\{m_{1: K, 1: L}\right\} \in\{0,1\}^{K \times L}$ where $m_{k, l}=0$ if $x_{k, l}$ is missing, and $m_{k, l}=1$ if $x_{k, l}$ is observed. We assume time intervals between two consecutive data entries can be different, and define the timestamps of the time series as $\mathbf{s}=\left\{s_{1: L}\right\} \in \mathbb{R}^L$. In summary, each time series is expressed as $\{\mathbf{X}, \mathbf{M}, \mathbf{s}\}$.

Probabilistic time series imputation is the task of estimating the distribution of the missing values of $\mathbf{X}$ by exploiting the observed values of $\mathbf{X}$. We note that this definition of imputation includes other related tasks, such as interpolation, which imputes all features at target time points, and forecasting, which imputes all features at future time points.

## (2) Denoising diffusion probabilistic models

Let us consider learning a model distribution $p_\theta\left(\mathbf{x}_0\right)$ that approximates a data distribution $q\left(\mathbf{x}_0\right)$. Let $\mathbf{x}_t$ for $t=1, \ldots, T$ be a sequence of latent variables in the same sample space as $\mathbf{x}_0$, which is denoted as $\mathcal{X}$. Diffusion probabilistic models [26] are latent variable models that are composed of two processes: the forward process and the reverse process. The forward process is defined by the following Markov chain:
$$
q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right):=\prod_{t=1}^T q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right) \text { where } q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right):=\mathcal{N}\left(\sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}\right)
$$
and $\beta_t$ is a small positive constant that represents a noise level. Sampling of $\mathbf{x}_t$ has the closed-form written as $q\left(\mathbf{x}_t \mid \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{\alpha_t} \mathbf{x}_0,\left(1-\alpha_t\right) \mathbf{I}\right)$ where $\hat{\alpha}_t:=1-\beta_t$ and $\alpha_t:=\prod_{i=1}^t \hat{\alpha}_i$. Then, $\mathbf{x}_t$ can be expressed as $\mathbf{x}_t=\sqrt{\alpha_t} \mathbf{x}_0+\left(1-\alpha_t\right) \boldsymbol{\epsilon}$ where $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. On the other hand, the reverse process denoises $\mathbf{x}_t$ to recover $\mathbf{x}_0$, and is defined by the following Markov chain:
$$
\begin{aligned}
& p_\theta\left(\mathbf{x}_{0: T}\right):=p\left(\mathbf{x}_T\right) \prod_{t=1}^T p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right), \quad \mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \\
& p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right):=\mathcal{N}\left(\mathbf{x}_{t-1} ; \boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right), \sigma_\theta\left(\mathbf{x}_t, t\right) \mathbf{I}\right)
\end{aligned}
$$
Ho et al. [11] has recently proposed denoising diffusion probabilistic models (DDPM), which considers the following specific parameterization of $p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)$ :
$$
\boldsymbol{\mu}_\theta\left(\mathbf{x}_t, t\right)=\frac{1}{\alpha_t}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\alpha_t}} \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)\right), \sigma_\theta\left(\mathbf{x}_t, t\right)=\tilde{\beta}_t^{1 / 2} \text { where } \tilde{\beta}_t= \begin{cases}\frac{1-\alpha_{t-1}}{1-\alpha_t} \beta_t & t>1 \\ \beta_1 & t=1\end{cases}
$$
