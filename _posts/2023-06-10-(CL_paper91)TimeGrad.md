---
title: (paper 91) Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting
categories: [GAN, TS]
tags: []
excerpt: 2021
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting

<br>

![figure2](/assets/img/ts/img429.png)

## Contents

0. Abstract
1. 



<br>

# 0. Abstract

TimeGrad

- AR model for MTS probabilistic forecasting
  - samples from the data distribution at each time step by estimating its gradient. 
- Diffusion probabilistic models
- Learns gradients by optimizing a variational bound on the data likelihood
- Inference time ) converts white noise into a sample of the distribution

<br>

# 1. Introduction

 TimeGrad = **Autoregressive EBMs**

- to solve the multivariate probabilistic time series forecasting problem
- train a model with all the inductive biases of probabilistic TS forecasting



Autoregressive-EBM

- AR)  Good performance in extrapolation into the future
- EBM) Flexibility of EBMs as a general purpose high-dimensional distribution model

<br>

Setup

- Section 2) Notation & Detail of EBM
- Section 3) MTS probabilsitic problem & TimeGrad
- Section 4) Experiments

<br>

# 2. Diffusion Probabilistic Model

Notation

- $\mathbf{x}^0 \sim q_{\mathcal{X}}\left(\mathbf{x}^0\right)$ : Multivariate training vector 

  - input space $\mathcal{X}=\mathbb{R}^D$ 

- $p_\theta\left(\mathbf{x}^0\right)$ : PDF which aims to approximate $q_{\mathcal{X}}\left(\mathbf{x}^0\right)$ 

  ( + allows for easy sampling )

<br>

Diffusion models 

- Latent variable models of the form $p_\theta\left(\mathbf{x}^0\right):=\int p_\theta\left(\mathbf{x}^{0: N}\right) \mathrm{d} \mathbf{x}^{1: N}$, 
  - where $\mathbf{x}^1, \ldots, \mathbf{x}^N$ are latents of dimension $\mathbb{R}^D$. 
- Unlike VAE, approximate posterior $q\left(\mathbf{x}^{1: N} \mid \mathbf{x}^0\right)$ is not trainable, but fixed to Markov chain ( = forward process )
  - $q\left(\mathbf{x}^{1: N} \mid \mathbf{x}^0\right)=\Pi_{n=1}^N q\left(\mathbf{x}^n \mid \mathbf{x}^{n-1}\right)$.
    - $q\left(\mathbf{x}^n \mid \mathbf{x}^{n-1}\right):=\mathcal{N}\left(\mathbf{x}^n ; \sqrt{1-\beta_n} \mathbf{x}^{n-1}, \beta_n \mathbf{I}\right) $.
    - Forward process uses an increasing variance schedule $\beta_1, \ldots, \beta_N$ with $\beta_n \in(0,1)$.

<br>

Reverser Process

- joint distn $p_\theta\left(\mathbf{x}^{0: N}\right)$
- also defined as a Markov chain 
  - with learned Gaussian transitions starting with $p\left(\mathbf{x}^N\right)=\mathcal{N}\left(\mathbf{x}^N ; \mathbf{0}, \mathbf{I}\right)$
- $p_\theta\left(\mathbf{x}^{0: N}\right):=p\left(\mathbf{x}^N\right) \Pi_{n=N}^1 p_\theta\left(\mathbf{x}^{n-1} \mid \mathbf{x}^n\right)$.
  - $p_\theta\left(\mathbf{x}^{n-1} \mid \mathbf{x}^n\right):=\mathcal{N}\left(\mathbf{x}^{n-1} ; \mu_\theta\left(\mathbf{x}^n, n\right), \Sigma_\theta\left(\mathbf{x}^n, n\right) \mathbf{I}\right)$.

<br>

Both $\mu_\theta: \mathbb{R}^D \times \mathbb{N} \rightarrow \mathbb{R}^D$ and $\Sigma_\theta: \mathbb{R}^D \times \mathbb{N} \rightarrow \mathbb{R}^{+}$take two inputs

- (1) Variable $\mathbf{x}^n \in \mathbb{R}^D$ 
- (2) Noise index $n \in \mathbb{N}$. 

<br>

Goal of $p_\theta\left(\mathbf{x}^{n-1} \mid \mathbf{x}^n\right)$ 

= eliminate the Gaussian noise added 

= $\theta$ are learned to fit the data distribution $q_{\mathcal{X}}\left(\mathbf{x}^0\right)$ 

- by minimizing the NLL via a variational bound

$\begin{array}{r}
\min _\theta \mathbb{E}_{q\left(\mathbf{x}^0\right)}\left[-\log p_\theta\left(\mathbf{x}^0\right)\right] \leq 
\min _\theta \mathbb{E}_{q\left(\mathbf{x}^{0: N}\right)}\left[-\log p_\theta\left(\mathbf{x}^{0: N}\right)+\log q\left(\mathbf{x}^{1: N} \mid \mathbf{x}^0\right)\right] 
\end{array}$.

<br>

Summary ( shown by (Ho et al., 2020) )

$q\left(\mathbf{x}^n \mid \mathbf{x}^0\right)=\mathcal{N}\left(\mathbf{x}^n ; \sqrt{\bar{\alpha}_n} \mathbf{x}^0,\left(1-\bar{\alpha}_n\right) \mathbf{I}\right) $.

- $\alpha_n:=1-\beta_n$.
- $\bar{\alpha}_n:=\Pi_{i=1}^n \alpha_i$.

<br>

file:///Users/LSH/Downloads/2101.12072.pdf
