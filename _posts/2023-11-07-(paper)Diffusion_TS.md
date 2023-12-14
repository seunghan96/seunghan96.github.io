---
title: Diffusion Models for Time Series Applications; A Survey
categories: [TS, GAN]
tags: []
excerpt: 2023
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Diffusion Models for Time Series Applications: A Survey

<br>

![figure2](/assets/img/ts/img497.png)

# Contents

0. Abstract

0. 




<br>


# Abstract

Diffusion model

- increasingly prominent in generative AI
- widely used in image, video, and text synthesis
- has been extended to TS

<br>

Primarily focus on diffusion-based methods for ...

- time series forecasting
- time series imputation
- time series generation

<br>

# 1. Introduction

Diffusion models: various real-world applications 

- Image synthesis (Austin et al., 2021; Dhariwal and Nichol, 2021; Ho et al., 2022a)
- Video generation (Harvey et al., 2022; Ho et al., 2022b; Yang et al., 2022b)
- Natural language processing (Li et al., 2022; Nikolay et al., 2022; Yu et al., 2022)
- Time series prediction (Rasul et al., 2021a; Li et al., 2022; Alcaraz and Strodthoff, 2023)

<br>

Diffusion models to TS

- (1) TS forecasting (Rasul et al., 2021a; Li et al., 2022; Biloš et al., 2022)
- (2) TS imputation (Tashiro et al., 2021; Alcaraz and Strodthoff, 2023; Liu et al., 2023)
- (3) TS generation (Lim et al., 2023)
  - aims to produce more TS samples with similar characteristics 

<br>

Diffusion-based methods for TS applications :

$\rightarrow$ developed from 3 fundamental formulations

- (1) Denoising diffusion probabilistic models (DDPMs)
- (2) Score-based generative models (SGMs)
- (3) Stochastic differential equations (SDEs)

<br>

Target distributions learned by the diffusion components in different methods often involve ***the condition on previous time steps***

<br>

However, design of the diffusion and denoising processes varies with different objectives of different tasks!

<br>

# 2. Basic of Diffusion Models

## (1) DDPM

Forward

- $q\left(\boldsymbol{x}^k \mid \boldsymbol{x}^{k-1}\right)=\mathcal{N}\left(\boldsymbol{x}^k ; \sqrt{\alpha_k} \boldsymbol{x}^{k-1},\left(1-\alpha_k\right) \boldsymbol{I}\right)$.
- $q\left(\boldsymbol{x}^k \mid \boldsymbol{x}^0\right)=\mathcal{N}\left(\boldsymbol{x}^k ; \sqrt{\tilde{\alpha_k}} \boldsymbol{x}^0,\left(1-\tilde{\alpha_k}\right) \boldsymbol{I}\right)$.

<br>

Backward

- $p_{\boldsymbol{\theta}}\left(\boldsymbol{x}^{k-1} \mid \boldsymbol{x}^k\right)=\mathcal{N}\left(\boldsymbol{x}^{k-1} ; \boldsymbol{\mu}_{\boldsymbol{\theta}}\left(\boldsymbol{x}^k, k\right), \boldsymbol{\Sigma}_{\boldsymbol{\theta}}\left(\boldsymbol{x}^k, k\right)\right)$.

<br>

Loss function:

- $\mathbb{E}_{q\left(\boldsymbol{x}^{0: K}\right)}\left[-\log p\left(\boldsymbol{x}^K\right)-\sum_{k=1}^K \log \frac{p_{\boldsymbol{\theta}}\left(\boldsymbol{x}^{k-1} \mid \boldsymbol{x}^k\right)}{q\left(\boldsymbol{x}^k \mid \boldsymbol{x}^{k-1}\right)}\right]$.

<br>

DDPM:

- (1) Simplify covariance matrix $\boldsymbol{\Sigma}_{\boldsymbol{\theta}}\left(\boldsymbol{x}^k, k\right)$ as a constant-dependent matrix $\sigma_k^2 \boldsymbol{I}$, 
- (2) Rewrite the mean as a function of a learnable noise term as
  - $\boldsymbol{\mu}_{\boldsymbol{\theta}}\left(\boldsymbol{x}^k, k\right)=\frac{1}{\sqrt{\alpha_k}}\left(\boldsymbol{x}^k-\zeta(k) \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\boldsymbol{x}^k, k\right)\right)$.
    - $\zeta(k)=\frac{1-\alpha_k}{\sqrt{1-\tilde{\alpha}_k}}$, 
    - $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$ : noise-matching network
  - $\mathbb{E}_{k, \boldsymbol{x}^0, \boldsymbol{\epsilon}}\left[\delta(k)\left\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\sqrt{\tilde{\alpha}_k} \boldsymbol{x}^0+\sqrt{1-\tilde{\alpha}_k} \boldsymbol{\epsilon}, k\right)\right\|^2\right]$.
    - where $\delta(k)=\frac{\left(1-\alpha_k\right)^2}{2 \sigma_k^2 \alpha_k\left(1-\tilde{\alpha}_k\right)}$ is a positive-valued weight
- (3) Discard $\delta(k)$

<br>

## (2) Score-based Generative Models

Consist of two modules

- (1) Score matching
- (2) Annealed Langevin dynamics (ALD)
  - sampling algorithm that generates samples with an iterative process by applying Langevin Monte Carlo at each update step

<br>

Stein score :

- score of a density function $q(\boldsymbol{x})$ = $\nabla_{\boldsymbol{x}} \log q(\boldsymbol{x})$

Score matching 

- approximate the Stein score with a score-matching network

<br>

### Denoising Score Matching

Process the observed data with the forward transition kernel

-  $q\left(\boldsymbol{x}^k \mid \boldsymbol{x}^0\right)=\mathcal{N}\left(\boldsymbol{x}^k ; \boldsymbol{x}^0, \sigma_k^2 \boldsymbol{I}\right)$.

Jointly estimate the Stein scores for the noise density distributions $q_{\sigma_1}(\boldsymbol{x}), q_{\sigma_2}(\boldsymbol{x}), \ldots, q_{\sigma_k}(\boldsymbol{x})$ 

Stein score is approximated by $\boldsymbol{s}_{\boldsymbol{\theta}}\left(\boldsymbol{x}, \sigma_k\right)$

<br>

**[ Training ]**

Initial objective function :

- $\mathbb{E}_{k, \boldsymbol{x}^0, \boldsymbol{x}^k}\left[\left\|\boldsymbol{s}_{\boldsymbol{\theta}}\left(\boldsymbol{x}^k, k\right)-\nabla_{\boldsymbol{x}^k} \log q_{\sigma_k}\left(\boldsymbol{x}^k\right)\right\|\right] $.

<br>

Tractable version of the objective function :

- $\mathbb{E}_{k, \boldsymbol{x}^0, \boldsymbol{x}^k}\left[\delta(k)\left\|\boldsymbol{s}_{\boldsymbol{\theta}}\left(\boldsymbol{x}^k, \sigma_k\right)+\frac{\boldsymbol{x}^k-\boldsymbol{x}^0}{\sigma_k^2}\right\|^2\right]$.
  - where $\delta(k)$ is a positive-valued weight depending on the noise scale $\sigma_k$.

<br>

**[ Inference ]**

After the score-matching network $\boldsymbol{s}_{\boldsymbol{\theta}}$ is learned, use ALD algorithm for sampling

- initialized with a sequence of increasing noise levels $\sigma_1, \ldots, \sigma_K$ and a starting point $\boldsymbol{x}^{K, 0} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})$. 

<br>

For $k=K, K-1, \ldots, 0, \boldsymbol{x}^k$ will be updated with $N$ iterations that compute

$\begin{aligned}
\boldsymbol{z} & \leftarrow \mathcal{N}(\mathbf{0}, \boldsymbol{I}) \\
\boldsymbol{x}^{k, n} & \leftarrow \boldsymbol{x}^{k, n-1}+\frac{1}{2} \eta_k \boldsymbol{s}_{\boldsymbol{\theta}}\left(\boldsymbol{x}^{k, n-1}, \sigma_k\right)+\sqrt{\eta_k} \boldsymbol{z}
\end{aligned}$.

<br>

## (3) SDE

pass

<br>

# 3. TS Forecasting

In recent years, **generative models** have been implemented for MTS forecasting

- ex) WaveNet : generative model with dilated causal convolutions
- ex) Conditional Normalizing Flow (Rasul et al. (2021b)) : Model MTS wiith an autoregressive deep learning model
  - data distribution is expressed by a conditional normalizing flow

<br>

Nevertheless, the common shortcoming of these models is that the ***functional structure of their target distributions are strictly constrained***

$\leftrightarrow$ Diffusion-based methods: can provide a less restrictive solution

<br>

## (1) Problem Formulation

Notation

- MTS: $\boldsymbol{X}^0=\left\{\boldsymbol{x}_1^0, \boldsymbol{x}_2^0, \ldots, \boldsymbol{x}_T^0 \mid \boldsymbol{x}_i^0 \in \mathbb{R}^D\right\}$
- Input: $\boldsymbol{X}_c^0=\left\{\boldsymbol{x}_1^0, \boldsymbol{x}_2^0, \ldots, \boldsymbol{x}_{t_0-1}^0\right\} $  ( = context window / condition / ... )
- Target: $\boldsymbol{X}_p^0=\left\{\boldsymbol{x}_{t_0}^0, \boldsymbol{x}_{t_0+1}^0, \ldots, \boldsymbol{x}_T^0\right\}$  ( = prediction interval )

<br>

In diffusion-based models, the problem is formulated as learning the **joint probabilistic distribution of data** in the prediction interval

- $q\left(\boldsymbol{x}_{t_0: T}^0 \mid \boldsymbol{x}_{1: t_0-1}^0\right)=\prod_{t=t_0}^T q\left(\boldsymbol{x}_t^0 \mid \boldsymbol{x}_{1: t_0-1}^0\right) $.

- $q\left(\boldsymbol{x}_{t_0: T}^0 \mid \boldsymbol{x}_{1: t_0-1}^0, \boldsymbol{c}_{1: T}\right)=\prod_{t=t_0}^T q\left(\boldsymbol{x}_t^0 \mid \boldsymbol{x}_{1: t_0-1}^0, \boldsymbol{c}_{1: T}\right)$ if covariate exists

<br>

Training

- randomly sample the context window followed by the prediction window
- can be seen as applying a moving window with size $T$ on the whole timeline

<br>

## (2) TimeGrad (Rasul et al. (2021a))

- Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting

- https://arxiv.org/abs/2101.12072
- https://seunghan96.github.io/gan/ts/(CL_paper91)TimeGrad/

<br>

First noticeable work on diffusion-based forecasting

<br>

Procedure

- step 1) injects noises to data at each **predictive time point**
- step 2) gradually denoise, **conditioned on historical time series**
  - use RNN to encode historical information

<br>

Conditional distribution:

- $\prod_{t=t_0}^T p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_t^0 \mid \boldsymbol{h}_{t-1}\right)$, where $\boldsymbol{h}_t=\mathrm{RNN}_{\boldsymbol{\theta}}\left(\boldsymbol{x}_t^0, \boldsymbol{c}_t, \boldsymbol{h}_{t-1}\right)$.

<br>

Objective function: NLL

- $\sum_{t=t_0}^T-\log p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_t^0 \mid \boldsymbol{h}_{t-1}\right)$ 
- $-\log p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_t^0 \mid \boldsymbol{h}_{t-1}\right)$ is upper bounded by $\mathbb{E}_{k, \boldsymbol{x}_t^0, \boldsymbol{\epsilon}}\left[\delta(k)\left\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\sqrt{\tilde{\alpha}_k} \boldsymbol{x}_t^0+\sqrt{1-\tilde{\alpha}_k} \boldsymbol{\epsilon}, \boldsymbol{h}_{t-1}, k\right)\right\|^2\right]$
  - very similar to standard diffusion, except for the ***inclusion of hidden states to represent the historical information***

<br>

Inference 

( in a step-by-step manner )

( suppose that the last time point of the complete time series is $\tilde{T}$ )

- Step 1) Derive the hidden state $\boldsymbol{h}_{\tilde{T}}$ 

  - based on the last available context window

- Step 2) Observation for the next time point $\tilde{T}+1$ is predicted in a similar way

  - $\boldsymbol{x}_{\tilde{T}+1}^k \leftarrow \frac{\left(\boldsymbol{x}_{\tilde{T}+1}^{k+1}-\zeta(k+1) \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\boldsymbol{x}_{\tilde{T}+1}^{k+1}, \boldsymbol{h}_{\tilde{T}}, k+1\right)\right)}{\sqrt{\alpha_{k+1}}}+\sigma_{k+1} \boldsymbol{z}$.

  - predicted $\boldsymbol{x}_{\tilde{T}+1}^k$ should be fed back to the RNN module to obtain $\boldsymbol{h}_{\tilde{T}+1}$ 

<br>

## (3) ScoreGrad

- https://arxiv.org/pdf/2106.10121.pdf

- ScoreGrad: Multivariate Probabilistic Time Series Forecasting with Continuous Energy-based Generative Models

***shares the same target distribution as TimeGrad, but it is alternatively built upon SDEs, extending the diffusion process from discrete to continuous and replacing the number of diffusion steps with an interval of integration***

<br>

Composed of ...

- (1) **Feature extraction module**
- (2) **Conditional SDE-based score-matching module**

<br>

(1) **Feature extraction module**
- almost identical to $h_t$ in TimeGrad
- use RNN/TCN/Attention ... all OK ( default: RNN )

<br>

(2) **Conditional SDE-based score-matching module**

- diffusion process is conducted through the same SDE
- associated time-reverse SDE is refined as following:
  - $\mathrm{d} \boldsymbol{x}_t=\left[f\left(\boldsymbol{x}_t, k\right)-g(k)^2 \nabla_{\boldsymbol{x}_t} \log q_k\left(\boldsymbol{x}_t \mid \boldsymbol{h}_t\right)\right] \mathrm{d} k+g(k) \mathrm{d} \boldsymbol{w}$.
    - $k \in[0, K]$: SDE integral time.

- Conditional score function $\nabla_{\boldsymbol{x}_t} \log q_k\left(\boldsymbol{x}_t \mid \boldsymbol{h}_t\right)$ : approximated with $\boldsymbol{s}_{\boldsymbol{\theta}}\left(\boldsymbol{x}_t^k, \boldsymbol{h}_t, k\right)$
- Inspired by WaveNet & DiffWave, $s_{\theta}$ : 8 connected residual blocks
  - each block: a bidirectional dilated convolution module, a gated activation unit, a skip-connection process, and an 1D CNN


<br>

### Objective Function

$\sum_{t=t_0}^T L_t(\boldsymbol{\theta})$, where $L_t(\boldsymbol{\theta})=\mathbb{E}_{k, \boldsymbol{x}_t^0, \boldsymbol{x}_t^k}\left[\delta(k)\left\|\boldsymbol{s}_{\boldsymbol{\theta}}\left(\boldsymbol{x}_t^k, \boldsymbol{h}_t, k\right)-\nabla_{\boldsymbol{x}_t} \log q_{0 k}\left(\boldsymbol{x}_t \mid \boldsymbol{x}_t^0\right)\right\|^2\right]$

only use the general expression of SDE 

- decide the specific type of SDE to use .... options : VE SDE, VP SDE, and sub-VP SDE 

<br>

### Sampling

use predictor-corrector sampler to sample from the time-reverse SDE.

<br>

## (4) $D^3VAE$ (Li et al. (2022))

(In practice) Insufficient observations

$D^3VAE$ : address the problem of limited and noisy TS

- employs a **coupled diffusion process** for data augmentation
- uses a **bidirectional auto-encoder (BVAE)** together with **denoising score matching** to clear the noise. 
- also considers **disentangling latent variables** by minimizing the overall correlation for better interpretability and stability of predictions. 

<br>

Notation

- Assumption: $q\left(\boldsymbol{Z} \mid \boldsymbol{x}_{1: t_0-1}^0\right)$.
- Conditional distribution of $\boldsymbol{Z}$ : approximate with $p_{\boldsymbol{\phi}}\left(\boldsymbol{Z} \mid \boldsymbol{x}_{1: t_0-1}^0\right)$ 
- Inference: can be generated from $p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_{t_0: T}^0 \mid \boldsymbol{Z}\right)$. 

$\rightarrow$ Prediction window is predicted based on the **context window ** however with  **latent variables $\boldsymbol{Z}$ as an intermediate **

<br>

Coupled diffusion process

- inject noises separately into the (1) context window and the (2) prediction window

- TimeGrad : injects noises to the observation at each time point individually

- Coupled diffusion process : applied to the whole period. 

  - For context window....
    - $\boldsymbol{x}_{1: t_0-1}^k=\sqrt{\tilde{\alpha}_k} \boldsymbol{x}_{1: t_0-1}^0+\sqrt{1-\tilde{\alpha}_k} \boldsymbol{\epsilon}$.

  - For prediction window ...

    - with adjusted noise levels $\alpha_k^{\prime}>\alpha_k$. 
    - $\boldsymbol{x}_{t_0: T}^k=\sqrt{\tilde{\alpha}_k^{\prime}} \boldsymbol{x}_{t_0: T}^0+\sqrt{1-\tilde{\alpha}_k^{\prime}} \boldsymbol{\epsilon}$.

  - simultaneously augments the context window and the prediction window

    $\rightarrow \therefore$ improve the generalization ability for short TS forecasting.

<br>

Backward process: 2 steps.

- step 1) Predict $\boldsymbol{x}_{t_0: T}^k$ with a BVAE 

  - composed of an encoder and a decoder with multiple residual blocks

    & takes the disturbed context window $\boldsymbol{x}_{1: t_0-1}^k$ as input

  - latent variables in $Z$ are gradually generated & fed into the model in a summation manner

  - output: $\hat{\boldsymbol{x}}_{t_0: T}^k$

    - predicted disturbed prediction window

- step 2) Cleaning of the predicted data with a denoising score matching module

  - obtained via a single-step gradient jump
    - $\hat{\boldsymbol{x}}_{t_0: T}^0 \leftarrow \hat{\boldsymbol{x}}_{t_0: T}^k-\sigma_0^2 \nabla_{\hat{\boldsymbol{x}}_{t_0: T}^k} E\left(\hat{\boldsymbol{x}}_{t_0: T}^k ; e\right)$.
    - where $\sigma_0$ is prescribed and $E\left(\hat{\boldsymbol{x}}_{t_0: T}^k ; e\right)$ is the energy function.

<br>

Disentanglement of latent variables $\boldsymbol{Z}$ 

- can efficiently enhance the model interpretability and reliability for prediction
- measured by the total correlation of $Z$

<br>

Objective Function

- $w_1 D_{K L}\left(q\left(\boldsymbol{x}_{t_0: T}^k\right) \| p_{\boldsymbol{\theta}}\left(\hat{\boldsymbol{x}}_{t_0: T}^k\right)\right)+w_2 \mathcal{L}_{D S M}+w_3 \mathcal{L}_{T C}+\mathcal{L}_{M S E}$.

<br>

## (5) DSPD

