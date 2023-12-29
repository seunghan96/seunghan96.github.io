---
title: Diffusion Models for Time Series Applications; A Survey
categories: [TS, GAN]
tags: []
excerpt: 2023
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Diffusion Models for Time Series Applications: A Survey

<br>

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

$$\rightarrow$$ developed from 3 fundamental formulations

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

- $$q\left(\boldsymbol{x}^k \mid \boldsymbol{x}^{k-1}\right)=\mathcal{N}\left(\boldsymbol{x}^k ; \sqrt{\alpha_k} \boldsymbol{x}^{k-1},\left(1-\alpha_k\right) \boldsymbol{I}\right)$$.
- $$q\left(\boldsymbol{x}^k \mid \boldsymbol{x}^0\right)=\mathcal{N}\left(\boldsymbol{x}^k ; \sqrt{\tilde{\alpha_k}} \boldsymbol{x}^0,\left(1-\tilde{\alpha_k}\right) \boldsymbol{I}\right)$$.

<br>

Backward

- $$p_{\boldsymbol{\theta}}\left(\boldsymbol{x}^{k-1} \mid \boldsymbol{x}^k\right)=\mathcal{N}\left(\boldsymbol{x}^{k-1} ; \boldsymbol{\mu}_{\boldsymbol{\theta}}\left(\boldsymbol{x}^k, k\right), \boldsymbol{\Sigma}_{\boldsymbol{\theta}}\left(\boldsymbol{x}^k, k\right)\right)$$.

<br>

Loss function:

- $$\mathbb{E}_{q\left(\boldsymbol{x}^{0: K}\right)}\left[-\log p\left(\boldsymbol{x}^K\right)-\sum_{k=1}^K \log \frac{p_{\boldsymbol{\theta}}\left(\boldsymbol{x}^{k-1} \mid \boldsymbol{x}^k\right)}{q\left(\boldsymbol{x}^k \mid \boldsymbol{x}^{k-1}\right)}\right]$$.

<br>

DDPM:

- (1) Simplify covariance matrix $$\boldsymbol{\Sigma}_{\boldsymbol{\theta}}\left(\boldsymbol{x}^k, k\right)$$ as a constant-dependent matrix $$\sigma_k^2 \boldsymbol{I}$$, 
- (2) Rewrite the mean as a function of a learnable noise term as
  - $$\boldsymbol{\mu}_{\boldsymbol{\theta}}\left(\boldsymbol{x}^k, k\right)=\frac{1}{\sqrt{\alpha_k}}\left(\boldsymbol{x}^k-\zeta(k) \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\boldsymbol{x}^k, k\right)\right)$$.
    - $$\zeta(k)=\frac{1-\alpha_k}{\sqrt{1-\tilde{\alpha}_k}}$$, 
    - $$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$$ : noise-matching network
  - $$\mathbb{E}_{k, \boldsymbol{x}^0, \boldsymbol{\epsilon}}\left[\delta(k) \mid \mid \boldsymbol{\epsilon}-\boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\sqrt{\tilde{\alpha}_k} \boldsymbol{x}^0+\sqrt{1-\tilde{\alpha}_k} \boldsymbol{\epsilon}, k\right) \mid \mid ^2\right]$$.
    - where $$\delta(k)=\frac{\left(1-\alpha_k\right)^2}{2 \sigma_k^2 \alpha_k\left(1-\tilde{\alpha}_k\right)}$$ is a positive-valued weight
- (3) Discard $$\delta(k)$$

<br>

## (2) Score-based Generative Models

Consist of two modules

- (1) Score matching
- (2) Annealed Langevin dynamics (ALD)
  - sampling algorithm that generates samples with an iterative process by applying Langevin Monte Carlo at each update step

<br>

Stein score :

- score of a density function $$q(\boldsymbol{x})$$ = $$\nabla_{\boldsymbol{x}} \log q(\boldsymbol{x})$$

Score matching 

- approximate the Stein score with a score-matching network

<br>

### Denoising Score Matching

Process the observed data with the forward transition kernel

-  $$q\left(\boldsymbol{x}^k \mid \boldsymbol{x}^0\right)=\mathcal{N}\left(\boldsymbol{x}^k ; \boldsymbol{x}^0, \sigma_k^2 \boldsymbol{I}\right)$$.

Jointly estimate the Stein scores for the noise density distributions $$q_{\sigma_1}(\boldsymbol{x}), q_{\sigma_2}(\boldsymbol{x}), \ldots, q_{\sigma_k}(\boldsymbol{x})$$ 

Stein score is approximated by $$\boldsymbol{s}_{\boldsymbol{\theta}}\left(\boldsymbol{x}, \sigma_k\right)$$

<br>

**[ Training ]**

Initial objective function :

- $$\mathbb{E}_{k, \boldsymbol{x}^0, \boldsymbol{x}^k}\left[ \mid \mid \boldsymbol{s}_{\boldsymbol{\theta}}\left(\boldsymbol{x}^k, k\right)-\nabla_{\boldsymbol{x}^k} \log q_{\sigma_k}\left(\boldsymbol{x}^k\right) \mid \mid \right] $$.

<br>

Tractable version of the objective function :

- $$\mathbb{E}_{k, \boldsymbol{x}^0, \boldsymbol{x}^k}\left[\delta(k) \mid \mid \boldsymbol{s}_{\boldsymbol{\theta}}\left(\boldsymbol{x}^k, \sigma_k\right)+\frac{\boldsymbol{x}^k-\boldsymbol{x}^0}{\sigma_k^2} \mid \mid ^2\right]$$.
  - where $$\delta(k)$$ is a positive-valued weight depending on the noise scale $$\sigma_k$$.

<br>

**[ Inference ]**

After the score-matching network $$\boldsymbol{s}_{\boldsymbol{\theta}}$$ is learned, use ALD algorithm for sampling

- initialized with a sequence of increasing noise levels $$\sigma_1, \ldots, \sigma_K$$ and a starting point $$\boldsymbol{x}^{K, 0} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})$$. 

<br>

For $$k=K, K-1, \ldots, 0, \boldsymbol{x}^k$$ will be updated with $$N$$ iterations that compute

$$\begin{aligned}
\boldsymbol{z} & \leftarrow \mathcal{N}(\mathbf{0}, \boldsymbol{I}) \\
\boldsymbol{x}^{k, n} & \leftarrow \boldsymbol{x}^{k, n-1}+\frac{1}{2} \eta_k \boldsymbol{s}_{\boldsymbol{\theta}}\left(\boldsymbol{x}^{k, n-1}, \sigma_k\right)+\sqrt{\eta_k} \boldsymbol{z}
\end{aligned}$$.

<br>

## (3) SDE

DDPMs and SGMs : Forward pass = ***Discrete process***

- (limitation) should carefully design the diffusion steps

<br>

Solution : Consider the diffusion process as ***continuous***

$$\rightarrow$$  Stochastic differential equation (SDE) (Song et al., 2021)

- Backward process : Time-reverse SDE
  - generate samples by solving this time-reverse SDE

<br>

General expression of SDE:

- $$\mathrm{d} \boldsymbol{x}=f(\boldsymbol{x}, k) \mathrm{d} k+g(k) \mathrm{d} \boldsymbol{w}$$.
  - $$\boldsymbol{w}$$ and $$\tilde{\boldsymbol{w}}$$ : Standard Wiener process and its time-reverse version
  - a continuous diffusion time $$k \in[0, K]$$. 

<br>

Time-reverse SDE ( Anderson (1982) )

- $$\mathrm{d} \boldsymbol{x}=\left[f(\boldsymbol{x}, k)-g(k)^2 \nabla_{\boldsymbol{x}} \log q_k(\boldsymbol{x})\right] \mathrm{d} k+g(k) \mathrm{d} \tilde{\boldsymbol{w}}$$.

<br>

### SDE $$\rightarrow$$ ODE

Sampling from the probability flow ODE as following has the same distribution as the time-reverse SDE:

$$\mathrm{d} \boldsymbol{x}=\left[f(\boldsymbol{x}, k)-\frac{1}{2} g(k)^2 \nabla_{\boldsymbol{x}} \log q_k(\boldsymbol{x})\right] \mathrm{d} k $$.

- $$f(\boldsymbol{x}, k)$$ : drift coefficient
- $$g(k)$$ : diffusion coefficient
- $$\nabla_{\boldsymbol{x}} \log q_k(\boldsymbol{x})$$ : Stein score
  - unknown but can be learned with a similar method as in SGMs with ...
    - $$\mathbb{E}_{k, \boldsymbol{x}^0, \boldsymbol{x}^k}\left[\delta(k) \mid \mid \boldsymbol{s}_{\boldsymbol{\theta}}\left(\boldsymbol{x}^k, k\right)-\nabla_{\boldsymbol{x}^k} \log q_{0 k}\left(\boldsymbol{x}^k \mid \boldsymbol{x}^0\right) \mid \mid ^2\right] $$.

<br>

How to write the diffusion processes of **DDPMs & SGMs** as **SDEs**?

- $$\alpha_k$$ : Parameter in DDPMs
- $$\sigma_k^2$$ : Noise level in SGMs. 

<br>

VP-SDE & VE-SDE

- VP-SDE: Variance Preserving (VP) SDE
  - SDE corresponding to DDPMs
  - $$\mathrm{d} \boldsymbol{x}=-\frac{1}{2} \alpha(k) \boldsymbol{x} \mathrm{d} k+\sqrt{\alpha(k)} \mathrm{d} \boldsymbol{w}$$.
    - $$\alpha(\cdot)$$ : continuous function
    - $$\alpha\left(\frac{k}{K}\right)=K\left(1-\alpha_k\right)$$ as $$K \rightarrow \infty$$. 

- VE-SDE: Variance Exploding (VE) SDE

  - SDE corresponding to SGMs

  - $$\mathrm{d} \boldsymbol{x}=\sqrt{\frac{\mathrm{d}\left[\sigma(k)^2\right]}{\mathrm{d} k}} \mathrm{~d} \boldsymbol{w}$$.
    - $$\sigma(\cdot)$$ : continuous function
    - $$\sigma\left(\frac{k}{K}\right)=\sigma_k$$ as $$K \rightarrow \infty$$ 

- Sub-VP SDE 

  - performs especially well on likelihoods, given by

    $$\mathrm{d} \boldsymbol{x}=-\frac{1}{2} \alpha(k) \boldsymbol{x} \mathrm{d} k+\sqrt{\alpha(k)\left(1-e^{-2 \int_0^k \alpha(s) \mathrm{d} s}\right)} \mathrm{d} \boldsymbol{w} $$.

<br>

### Summary

Objective function involves a perturbation distribution $$q_{0 k}\left(\boldsymbol{x}^k \mid \boldsymbol{x}^0\right)$$ that varies for different SDEs

$$q_{0 k}\left(\boldsymbol{x}^k \mid \boldsymbol{x}^0\right)= \begin{cases}\mathcal{N}\left(x^k ; x^0,\left[\sigma(k)^2-\sigma(0)^2\right] \boldsymbol{I}\right), & \text { (VP SDE) } \\ \mathcal{N}\left(x^k ; x^0 e^{-\frac{1}{2} \int_0^k \alpha(s) \mathrm{d} s},\left[1-e^{-\int_0^k \alpha(s) \mathrm{d} s}\right] \boldsymbol{I}\right), & \text { (VE SDE) } \\ \left.\mathcal{N}\left(x^k ; x^0 e^{-\frac{1}{2} \int_0^k \alpha(s) \mathrm{d} s},\left[1-e^{-\int_0^k \alpha(s) \mathrm{d} s}\right]\right]^2 \boldsymbol{I}\right), & \text { (sub-VP SDE) }\end{cases}$$.

- After successfully learning $$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}, k)$$, samples are produced by deriving the solutions to the time-reverse SDE or the probability flow ODE with techniques such as ALD.

<br>

# 3. TS Forecasting

In recent years, **generative models** have been implemented for MTS forecasting

- ex) WaveNet : generative model with dilated causal convolutions
- ex) Conditional Normalizing Flow (Rasul et al. (2021b)) : Model MTS wiith an autoregressive deep learning model
  - data distribution is expressed by a conditional normalizing flow

<br>

Nevertheless, the common shortcoming of these models is that the ***functional structure of their target distributions are strictly constrained***

$$\leftrightarrow$$ Diffusion-based methods: can provide a less restrictive solution

<br>

## (1) Problem Formulation

Notation

- MTS: $$\boldsymbol{X}^0=\left\{\boldsymbol{x}_1^0, \boldsymbol{x}_2^0, \ldots, \boldsymbol{x}_T^0 \mid \boldsymbol{x}_i^0 \in \mathbb{R}^D\right\}$$
- Input: $$\boldsymbol{X}_c^0=\left\{\boldsymbol{x}_1^0, \boldsymbol{x}_2^0, \ldots, \boldsymbol{x}_{t_0-1}^0\right\} $$  ( = context window / condition / ... )
- Target: $$\boldsymbol{X}_p^0=\left\{\boldsymbol{x}_{t_0}^0, \boldsymbol{x}_{t_0+1}^0, \ldots, \boldsymbol{x}_T^0\right\}$$  ( = prediction interval )

<br>

In diffusion-based models, the problem is formulated as learning the **joint probabilistic distribution of data** in the prediction interval

- $$q\left(\boldsymbol{x}_{t_0: T}^0 \mid \boldsymbol{x}_{1: t_0-1}^0\right)=\prod_{t=t_0}^T q\left(\boldsymbol{x}_t^0 \mid \boldsymbol{x}_{1: t_0-1}^0\right) $$.

- $$q\left(\boldsymbol{x}_{t_0: T}^0 \mid \boldsymbol{x}_{1: t_0-1}^0, \boldsymbol{c}_{1: T}\right)=\prod_{t=t_0}^T q\left(\boldsymbol{x}_t^0 \mid \boldsymbol{x}_{1: t_0-1}^0, \boldsymbol{c}_{1: T}\right)$$ if covariate exists

<br>

Training

- randomly sample the context window followed by the prediction window
- can be seen as applying a moving window with size $$T$$ on the whole timeline

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

- $$\prod_{t=t_0}^T p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_t^0 \mid \boldsymbol{h}_{t-1}\right)$$, where $$\boldsymbol{h}_t=\mathrm{RNN}_{\boldsymbol{\theta}}\left(\boldsymbol{x}_t^0, \boldsymbol{c}_t, \boldsymbol{h}_{t-1}\right)$$.

<br>

Objective function: NLL

- $$\sum_{t=t_0}^T-\log p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_t^0 \mid \boldsymbol{h}_{t-1}\right)$$ 
- $$-\log p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_t^0 \mid \boldsymbol{h}_{t-1}\right)$$ is upper bounded by $$\mathbb{E}_{k, \boldsymbol{x}_t^0, \boldsymbol{\epsilon}}\left[\delta(k) \mid \mid \boldsymbol{\epsilon}-\boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\sqrt{\tilde{\alpha}_k} \boldsymbol{x}_t^0+\sqrt{1-\tilde{\alpha}_k} \boldsymbol{\epsilon}, \boldsymbol{h}_{t-1}, k\right) \mid \mid ^2\right]$$
  - very similar to standard diffusion, except for the ***inclusion of hidden states to represent the historical information***

<br>

Inference 

( in a step-by-step manner )

( suppose that the last time point of the complete time series is $$\tilde{T}$$ )

- Step 1) Derive the hidden state $$\boldsymbol{h}_{\tilde{T}}$$ 

  - based on the last available context window

- Step 2) Observation for the next time point $$\tilde{T}+1$$ is predicted in a similar way

  - $$\boldsymbol{x}_{\tilde{T}+1}^k \leftarrow \frac{\left(\boldsymbol{x}_{\tilde{T}+1}^{k+1}-\zeta(k+1) \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\boldsymbol{x}_{\tilde{T}+1}^{k+1}, \boldsymbol{h}_{\tilde{T}}, k+1\right)\right)}{\sqrt{\alpha_{k+1}}}+\sigma_{k+1} \boldsymbol{z}$$.

  - predicted $$\boldsymbol{x}_{\tilde{T}+1}^k$$ should be fed back to the RNN module to obtain $$\boldsymbol{h}_{\tilde{T}+1}$$ 

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
- almost identical to $$h_t$$ in TimeGrad
- use RNN/TCN/Attention ... all OK ( default: RNN )

<br>

(2) **Conditional SDE-based score-matching module**

- diffusion process is conducted through the same SDE
- associated time-reverse SDE is refined as following:
  - $$\mathrm{d} \boldsymbol{x}_t=\left[f\left(\boldsymbol{x}_t, k\right)-g(k)^2 \nabla_{\boldsymbol{x}_t} \log q_k\left(\boldsymbol{x}_t \mid \boldsymbol{h}_t\right)\right] \mathrm{d} k+g(k) \mathrm{d} \boldsymbol{w}$$.
    - $$k \in[0, K]$$: SDE integral time.

- Conditional score function $$\nabla_{\boldsymbol{x}_t} \log q_k\left(\boldsymbol{x}_t \mid \boldsymbol{h}_t\right)$$ : approximated with $$\boldsymbol{s}_{\boldsymbol{\theta}}\left(\boldsymbol{x}_t^k, \boldsymbol{h}_t, k\right)$$
- Inspired by WaveNet & DiffWave, $$s_{\theta}$$ : 8 connected residual blocks
  - each block: a bidirectional dilated convolution module, a gated activation unit, a skip-connection process, and an 1D CNN


<br>

### Objective Function

$$\sum_{t=t_0}^T L_t(\boldsymbol{\theta})$$, where $$L_t(\boldsymbol{\theta})=\mathbb{E}_{k, \boldsymbol{x}_t^0, \boldsymbol{x}_t^k}\left[\delta(k) \mid \mid \boldsymbol{s}_{\boldsymbol{\theta}}\left(\boldsymbol{x}_t^k, \boldsymbol{h}_t, k\right)-\nabla_{\boldsymbol{x}_t} \log q_{0 k}\left(\boldsymbol{x}_t \mid \boldsymbol{x}_t^0\right) \mid \mid ^2\right]$$

only use the general expression of SDE 

- decide the specific type of SDE to use .... options : VE SDE, VP SDE, and sub-VP SDE 

<br>

### Sampling

use predictor-corrector sampler to sample from the time-reverse SDE.

<br>

## (4) $$D^3VAE$$ (Li et al. (2022))

(In practice) Insufficient observations

$$D^3VAE$$ : address the problem of limited and noisy TS

- employs a **coupled diffusion process** for data augmentation
- uses a **bidirectional auto-encoder (BVAE)** together with **denoising score matching** to clear the noise. 
- also considers **disentangling latent variables** by minimizing the overall correlation for better interpretability and stability of predictions. 

<br>

Notation

- Assumption: $$q\left(\boldsymbol{Z} \mid \boldsymbol{x}_{1: t_0-1}^0\right)$$.
- Conditional distribution of $$\boldsymbol{Z}$$ : approximate with $$p_{\boldsymbol{\phi}}\left(\boldsymbol{Z} \mid \boldsymbol{x}_{1: t_0-1}^0\right)$$ 
- Inference: can be generated from $$p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_{t_0: T}^0 \mid \boldsymbol{Z}\right)$$. 

$$\rightarrow$$ Prediction window is predicted based on the **context window ** however with  **latent variables $$\boldsymbol{Z}$$ as an intermediate **

<br>

Coupled diffusion process

- inject noises separately into the (1) context window and the (2) prediction window

- TimeGrad : injects noises to the observation at each time point individually

- Coupled diffusion process : applied to the whole period. 

  - For context window....
    - $$\boldsymbol{x}_{1: t_0-1}^k=\sqrt{\tilde{\alpha}_k} \boldsymbol{x}_{1: t_0-1}^0+\sqrt{1-\tilde{\alpha}_k} \boldsymbol{\epsilon}$$.

  - For prediction window ...

    - with adjusted noise levels $$\alpha_k^{\prime}>\alpha_k$$. 
    - $$\boldsymbol{x}_{t_0: T}^k=\sqrt{\tilde{\alpha}_k^{\prime}} \boldsymbol{x}_{t_0: T}^0+\sqrt{1-\tilde{\alpha}_k^{\prime}} \boldsymbol{\epsilon}$$.

  - simultaneously augments the context window and the prediction window

    $$\rightarrow \therefore$$ improve the generalization ability for short TS forecasting.

<br>

Backward process: 2 steps.

- step 1) Predict $$\boldsymbol{x}_{t_0: T}^k$$ with a BVAE 

  - composed of an encoder and a decoder with multiple residual blocks

    & takes the disturbed context window $$\boldsymbol{x}_{1: t_0-1}^k$$ as input

  - latent variables in $$Z$$ are gradually generated & fed into the model in a summation manner

  - output: $$\hat{\boldsymbol{x}}_{t_0: T}^k$$

    - predicted disturbed prediction window

- step 2) Cleaning of the predicted data with a denoising score matching module

  - obtained via a single-step gradient jump
    - $$\hat{\boldsymbol{x}}_{t_0: T}^0 \leftarrow \hat{\boldsymbol{x}}_{t_0: T}^k-\sigma_0^2 \nabla_{\hat{\boldsymbol{x}}_{t_0: T}^k} E\left(\hat{\boldsymbol{x}}_{t_0: T}^k ; e\right)$$.
    - where $$\sigma_0$$ is prescribed and $$E\left(\hat{\boldsymbol{x}}_{t_0: T}^k ; e\right)$$ is the energy function.

<br>

Disentanglement of latent variables $$\boldsymbol{Z}$$ 

- can efficiently enhance the model interpretability and reliability for prediction
- measured by the total correlation of $$Z$$

<br>

Objective Function

- $$w_1 D_{K L}\left(q\left(\boldsymbol{x}_{t_0: T}^k\right) \| p_{\boldsymbol{\theta}}\left(\hat{\boldsymbol{x}}_{t_0: T}^k\right)\right)+w_2 \mathcal{L}_{D S M}+w_3 \mathcal{L}_{T C}+\mathcal{L}_{M S E}$$.

<br>

## (5) DSPD

TS can be modelled as values from an underlying continuous function

- context window : $$\boldsymbol{X}_c^0=\left\{\boldsymbol{x}(1), \boldsymbol{x}(2), \ldots, \boldsymbol{x}\left(t_0-1\right)\right\}$$ 
- prediction window : $$\boldsymbol{X}_p^0=\left\{\boldsymbol{x}\left(t_0\right), \boldsymbol{x}\left(t_0+1\right), \ldots, \boldsymbol{x}(T)\right\}$$, 
  - where $$\boldsymbol{x}(\cdot)$$ is a continuous function of the time point $$t$$.

<br>

Diffusion are no longer applied to **VECTOR** observations at each time point. 

$$\leftrightarrow$$ To **continuous FUNCTION** $$\boldsymbol{x}(\cdot)$$

- which means noises will be injected and removed ***from a function rather than a vector***

<br>

## (6) DiffSTG

Spatio-temporal graphs (STGs) 

- encodes spatial and temporal relationships
- ex) traffic flow prediction (Li et al., 2018), weather forecasting (Simeunović et al., 2021), and finance prediction (Zhou et al., 2011)

- graph $$\mathcal{G}=\{\mathcal{V}, \mathcal{E}, \boldsymbol{W}\}$$, 

<br>

MTS are models as graph signals $$\boldsymbol{X}_c^0=\left\{\boldsymbol{x}_1^0, \boldsymbol{x}_2^0, \ldots, \boldsymbol{x}_{t_0-1}^0 \mid \boldsymbol{x}_t^0 \in \mathbb{R}^{D \times N}\right\}$$, 

- $$D$$-dimensional observations
- $$N$$ entities at each time point $$t$$. 

<br>

Goal: predict $$\boldsymbol{X}_p^0=\left\{\boldsymbol{x}_{t_0}^0, \boldsymbol{x}_{t_0+1}^0, \ldots, \boldsymbol{x}_T^0 \mid \boldsymbol{x}_t^0 \in \mathbb{R}^{D \times N}\right\}$$ based on $$\boldsymbol{X}_c$$. 

<br>

DiffSTG 

- Applies diffusion models on STG forecasting with a graph-based noise matching network called UGnet (Wen et al., 2023)
- ***extension of DDPM-based forecasting to STGs with an additional condition on the graph structure***
  - $$p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_{t_0: T}^0 \mid \boldsymbol{x}_{1: t_0-1}^0, \boldsymbol{W}\right) $$.

<br>

Loss Function:

$$\mathbb{E}_{k, \boldsymbol{x}_{t_0: T}^0, \boldsymbol{\epsilon}}\left[\delta(k) \mid \mid \boldsymbol{\epsilon}-\boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\sqrt{\tilde{\alpha}_k} \boldsymbol{x}_{t_0: T}^0+\sqrt{1-\tilde{\alpha}_k} \boldsymbol{\epsilon}, \boldsymbol{x}_{1: t_0-1}^0, k, \boldsymbol{W}\right) \mid \mid ^2\right] $$.

- treats the context window and the prediction window as samples from two separate sample spaces, 
  - $$\boldsymbol{X}_c^0 \in \mathcal{X}_c$$ and $$\boldsymbol{X}_p^0 \in \mathcal{X}_p$$ with $$\mathcal{X}_c$$ and $$\mathcal{X}_p$$ being two individual sample spaces
- but more reasonable to treat the two windows as a complete sample from the same sample space. 

<br>

$$\rightarrow$$ Wen et al. (2023) reformulate the forecasting problem and revise the approximation 

- by masking the future TS from the whole TS

- $$p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_{1: T}^0 \mid \boldsymbol{x}_{1: t_0-1}^0, \boldsymbol{W}\right)$$.

$$\mathbb{E}_{k, \boldsymbol{x}_{1: T}^0, \boldsymbol{\epsilon}}\left[\delta(k) \mid \mid \boldsymbol{\epsilon}-\boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\sqrt{\tilde{\alpha}_k} \boldsymbol{x}_{1: T}^0+\sqrt{1-\tilde{\alpha}_k} \boldsymbol{\epsilon}, \boldsymbol{x}_{1: t_0-1}^0, k, \boldsymbol{W}\right) \mid \mid ^2\right] $$.

<br>

Q) How to encode the **graph structural information** in the noise-matching network $$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$$ ? 

A) Wen et al. (2023) : proposed **UGnet**

- Unet-based network architecture + GNN
  - to process time dependency and spatial relationships simultaneously. 
- Input: $$\boldsymbol{x}_{1: T}^k, \boldsymbol{x}_{1: t_0-1}^0, k$$ and $$\boldsymbol{W}$$ 
- Output: prediction of the associated error $$\boldsymbol{\epsilon}$$.

<br>

## (7) GCRDD

Graph convolutional recurrent denoising diffusion model (GCRDD)

- diffusion-based model for STG forecasting (Li et al., 2023)

<br>

Difference btw DiffSTG

- uses  **"hidden states" from RNN** ( not raw value ) to store historical information as TimeGrad 
- employs a **different network structure for the noise-matching** term $$\boldsymbol{\epsilon}_\theta$$.

<br>

Approximates the target distribution with ...

$$\prod_{t=t_0}^T p_{\boldsymbol{\theta}}\left(\boldsymbol{x}_t^0 \mid h_{t-1}, \boldsymbol{W}\right)$$.

- where the hidden state is computed with a **graph-modified GRU**
  - $$\boldsymbol{h}_t=\operatorname{GraphGRU}_{\boldsymbol{\theta}}\left(\boldsymbol{x}_t^0, \boldsymbol{h}_{t-1}, \boldsymbol{W}\right) $$.

<br>

**Graph-modified GRU**

- weight matrix in traditional GRU $$\rightarrow$$ graph convolution
  - both temporal and spatial information is stored in $$h$$

<br>

Loss function

- a similar form of TimeGrad, but with additional graph structural info
- $$\mathbb{E}_{k, \boldsymbol{x}_t^0, \boldsymbol{\epsilon}}\left[\delta(k) \mid \mid \boldsymbol{\epsilon}-\boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\sqrt{\tilde{\alpha}_k} \boldsymbol{x}_t^0+\sqrt{1-\tilde{\alpha}_k} \boldsymbol{\epsilon}, \boldsymbol{h}_{t-1}, \boldsymbol{W}, k\right) \mid \mid ^2\right] $$.

<br>

Noise-matching term

- adopts a variant of DiffWave
- incorporates a graph convolution component to process spatial information in $$\boldsymbol{W}$$. 

<br>

Summary: GCRDD is the same as TimeGrad except that the sample generated at each time point is a **matrix** rather than a **vector**

<br>

# 4. TS Imputation

Most existing approaches:

- involve the RNN architecture to encode time-dependency

<br>

Probabilistic imputation models

- ex) GP-VAE (Fortuin et al., 2020) and V-RIN (Mulyadi et al., 2021) 
- shown their practical value in recent years

<br>

Diffusion models have also been applied to TS imputation tasks

-  (Tashiro et al., 2021; Alcaraz and Strodthoff, 2023)

- enjoys high flexibility in the assumption of the true data distribution

<br>

4 diffusion-based methods

- MTS imputation x 3
- STG imputation x1

<br>

## (1) Problem Formulation

Notation

- MTS $$\boldsymbol{X}^0=\left\{\boldsymbol{x}_1^0, \boldsymbol{x}_2^0, \ldots, \boldsymbol{x}_T^0 \mid \boldsymbol{x}_i^0 \in \mathbb{R}^D\right\}$$, where $$\boldsymbol{X}^0 \in \mathbb{R}^{D \times T}$$
  - Incomplete matrix of observations

- Goal: Predict the values of missing data

  - Observed data: $$\boldsymbol{X}_{o b}^0$$ 

  - Missing data: $$\boldsymbol{X}_{m s}^0$$. 

    ( Both $$\boldsymbol{X}_{o b}^0$$ and $$\boldsymbol{X}_{m s}^0$$ have the same dimension as $$\boldsymbol{X}^0$$ )

- Find conditional probability distribution $$q\left(\boldsymbol{X}_{m s}^0 \mid \boldsymbol{X}_{o b}^0\right)$$.

<br>

For practical purposes ....

- (1) zero padding is applied to the incomplete matrix $$\boldsymbol{X}^0$$ 
- (2) $$M \in \mathbb{R}^{D \times T}$$ : missingness indication matrix
  - 0: missing & 1: observed

<br>

### [ Training ]  

Fraction of the actually observed data in $$\boldsymbol{X}^0$$ is randomly selected to be the true values of missing data, and the rest of the observed data will be the condition for prediction.

- Training mask $$\boldsymbol{M}^{\prime} \in \mathbb{R}^{D \times T}$$ 
  - used to obtain $$\boldsymbol{X}_{o b}^0$$ and $$\boldsymbol{X}_{m s}^0$$. 
  - 0: missing & 1: observed

That is ...

- $$\boldsymbol{X}_{o b}^0=\boldsymbol{M}^{\prime} \odot \boldsymbol{X}^0$$.
- $$\boldsymbol{X}_{m s}^0=\left(\boldsymbol{M}-\boldsymbol{M}^{\prime}\right) \odot \boldsymbol{X}^0$$,



### [ Inference ]  

All actually observed data are used as the condition

( $$\boldsymbol{X}_{o b}^0=\boldsymbol{M} \odot \boldsymbol{X}^0$$ )

<br>

## (2) CSDI

Conditional Score-based Diffusion model for Imputation (CSDI)

- pioneering work on diffusion-based time series imputation (Tashiro et al., 2021)
- Identical to TimeGrad, the basic diffusion formulation of CSDI is also DDPM

<br>

TimeGrad vs. CSDI

- TimeGrad: historical information is encoded by an RNN

  $$\rightarrow$$ hampers the direct extension of TimeGrad to imputation tasks

  ( computation of hidden states may be interrupted by missing values in the context window )

- CSDI: applies the diffusion and reverse processes to the matrix of missing data, $$\boldsymbol{X}_{m s}^0$$. 

<br>

Reverse transition kernel ( of CSDI ) .... conditioned on $$\boldsymbol{X}_{o b}^0$$ 

- $$ p_{\boldsymbol{\theta}}\left(\boldsymbol{X}_{m s}^{k-1} \mid \boldsymbol{X}_{m s}^k, \boldsymbol{X}_{o b}^0\right)  =\mathcal{N}\left(\boldsymbol{X}_{m s}^{k-1} ; \boldsymbol{\mu}_{\boldsymbol{\theta}}\left(\boldsymbol{X}_{m s}^k, k \mid \boldsymbol{X}_{o b}^0\right), \sigma_{\boldsymbol{\theta}}\left(\boldsymbol{X}_{m s}^k, k \mid \boldsymbol{X}_{o b}^0\right) \boldsymbol{I}\right)$$.

  - $$\boldsymbol{\mu}_{\boldsymbol{\theta}}\left(\boldsymbol{X}_{m s}^k, k \mid \boldsymbol{X}_{o b}^0\right)=\frac{1}{\sqrt{\alpha_k}}\left(\boldsymbol{X}_{m s}^k-\zeta(k) \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\boldsymbol{X}_{m s}^k, k \mid \boldsymbol{X}_{o b}^0\right)\right)$$.
    - with $$\boldsymbol{X}_{m s}^k=\sqrt{\tilde{\alpha}_k} \boldsymbol{X}_{m s}^0+\sqrt{1-\tilde{\alpha}_k} \boldsymbol{\epsilon}$$. 

  - Variance : different from the version in DDPM
    - DDPM: some pre-specified constant $$\sigma_k$$ with $$k=1,2, \ldots, K$$ ( = variance as hyperparameter )
    - CSDI: learnable version $$\sigma_{\boldsymbol{\theta}}$$ with parameter $$\boldsymbol{\theta}$$. ..... but both ways are OK

<br>

Loss Function:

$$\mathbb{E}_{k, \boldsymbol{X}_{m s}^0, \boldsymbol{\epsilon}}\left[\delta(k) \mid \mid \boldsymbol{\epsilon}-\boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\sqrt{\tilde{\alpha}_k} \boldsymbol{X}_{m s}^0+\sqrt{1-\tilde{\alpha}_k} \boldsymbol{\epsilon}, \boldsymbol{X}_{o b}^0, k\right) \mid \mid ^2\right]$$.

<br>

Noise-matching network $$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$$ : adopts the DiffWave (Kong et al., 2021)

<br>

Inference ( = imputation  = sampling ) 

- by **generating the target matrix of missing values**  ( same way as DDPM )

- $$\boldsymbol{X}_{o b}^0$$ in the sampling process is identical to the zero padding version of $$\boldsymbol{X}^0$$, 
- Starting point of the sampling process : **random Gaussian imputation target** $$\boldsymbol{X}_{m s}^K \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})$$.
- For $$k=K-1, \ldots, 1$$, 
  - $$\boldsymbol{X}_{m s}^k \leftarrow \frac{\left(\boldsymbol{X}_{m s}^{k+1}-\zeta(k+1) \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\boldsymbol{X}_{m s}^{k+1}, \boldsymbol{X}_{o b}^0, k+1\right)\right)}{\sqrt{\alpha_{k+1}}}+\sigma_{\boldsymbol{\theta}} \boldsymbol{Z}$$.
    - where $$\boldsymbol{Z} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})$$ for $$k=K-1, \ldots, 1$$, and $$\boldsymbol{Z}=\mathbf{0}$$ for $$k=0$$

<br>

## (3) DSPD

pass

<br>

## (4) SSSD

Structured state space diffusion (SSSD)

- differs from the aforementioned two methods,

  in that it has the ***whole TS matrix $$\boldsymbol{X}^0$$ as the generative target*** in its diffusion module

<br>

Noise-matching network $$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$$,

- "structured state space diffusion"

  - adopts the state space model (Gu et al., 2022) as the internal architecture

- can also take other architectures 

  - ex) DiffWave-based network in CSDI (Tashiro et al., 2021)
  - ex) SaShiMi, a generative model for sequential data (Goel et al., 2022). 

  $$\rightarrow$$ However, the authors of ***SSSD have shown empirically that the structured state space model generally generates the best imputation***

<br>

### Training

Generative target of SSSD = **whole TS matrix** $$\boldsymbol{X}^0 \in \mathbb{R}^{D \times T}$$, 

( rather than a matrix that particularly represents the missing values. )

- also processed with zero padding

<br>

Conditional information:  $$\boldsymbol{X}_c^0=\operatorname{Concat}\left(\boldsymbol{X}^0 \odot \boldsymbol{M}_c, \boldsymbol{M}_c\right)$$,

- where $$\boldsymbol{M}_c$$ is a missingness indication matrix ( 1 = known )

<br>

### Loss function:  2 options

(v1) simple conditional variant of the DDPM objective function:

- $$\mathbb{E}_{k, \boldsymbol{X}^0, \boldsymbol{\epsilon}}\left[\delta(k) \mid \mid \boldsymbol{\epsilon}-\boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\sqrt{\tilde{\alpha}_k} \boldsymbol{X}^0+\sqrt{1-\tilde{\alpha}_k} \boldsymbol{\epsilon}, \boldsymbol{X}_c^0, k\right) \mid \mid ^2\right]$$.
  - where $$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$$ is structured state space model

(v2) computed with only known data

- $$\mathbb{E}_{k, \boldsymbol{X}^0, \boldsymbol{\epsilon}}\left[\delta(k) \mid \mid \boldsymbol{\epsilon} \odot \boldsymbol{M}_c-\boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\sqrt{\tilde{\alpha}_k} \boldsymbol{X}^0+\sqrt{1-\tilde{\alpha}_k} \boldsymbol{\epsilon}, \boldsymbol{X}_c^0, k\right) \odot \boldsymbol{M}_c \mid \mid ^2\right] $$.

- According to Alcaraz and Strodthoff (2023)... (v2) is better

<br>

### Inference

- Applies to the unknown entries in $$\boldsymbol{X}^0$$, namely, $$\left(1-\boldsymbol{M}_c\right) \odot \boldsymbol{X}^0$$.

<br>

Can also be applied to forecasting tasks

( = Future TS = long block of missing values on the right of $$\boldsymbol{X}^0$$ )

- still, underperform compared to Autoformer

<br>

## (5) PriSTI

Diffusion-based model for STG imputation (Liu et al., 2023)

<br>

Different from DiffSTG,  PriSTI is designed for STGs with **only one feature**

- Graph signal has the form $$\boldsymbol{X}^0=\left\{\boldsymbol{x}_1^0, \boldsymbol{x}_2^0, \ldots, \boldsymbol{x}_T^0\right\} \in \mathbb{R}^{N \times T}$$. 
  - $$\boldsymbol{x}_t^0 \in \mathbb{R}^N$$ : observed values of $$N$$ nodes at time point $$t$$. 
- ex) traffic prediction ( $$\mathrm{Li}$$ et al., 2018) , weather forecasting (Yi et al., 2016). 
  - METR-LA: STG dataset that contains traffic speed collected by 207 sensors
    - only one node attribute ( = traffic speed )
    - geographic relationship between different sensors is stored in the weighted adjacency matrix $$\boldsymbol{W}$$, 
  - number of nodes $$N$$ can be considered as the number of features $$D$$ in CSDI. 

<br>

Incorporates the **underlying relationship** between each pair of nodes in the **conditional info**

- Before) $$\boldsymbol{\mu}_{\boldsymbol{\theta}}\left(\boldsymbol{X}_{m s}^k, k \mid \boldsymbol{X}_{o b}^0\right)=\frac{1}{\sqrt{\alpha_k}}\left(\boldsymbol{X}_{m s}^k-\zeta(k) \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\boldsymbol{X}_{m s}^k, k \mid \boldsymbol{X}_{o b}^0\right)\right)$$
- After) $$\boldsymbol{\mu}_{\boldsymbol{\theta}}\left(\boldsymbol{X}_{m s}^k, k \mid \boldsymbol{X}_{o b}^0, \boldsymbol{W}\right)=\frac{1}{\sqrt{\alpha_k}}\left(\boldsymbol{X}_{m s}^k-\zeta(k) \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\boldsymbol{X}_{m s}^k, \boldsymbol{X}_{o b}^0, k, \boldsymbol{W}\right)\right)$$

<br>

### Loss function

$$\mathbb{E}_{k, \boldsymbol{X}_{m s}^0, \boldsymbol{\epsilon}}\left[\delta(k) \mid \mid \boldsymbol{\epsilon}-\boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\sqrt{\tilde{\alpha}_k} \boldsymbol{X}_{m s}^0+\sqrt{1-\tilde{\alpha}_k} \boldsymbol{\epsilon}, \boldsymbol{X}_{o b}^0, k, \boldsymbol{W}\right) \mid \mid ^2\right]$$.

- conditional information, $$\boldsymbol{X}_{o b}^0$$ is processed with linear interpolation
  - enhance the denoising capability 

<br>

Noise-matching network $$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$$ : composed of 2 modules

- **(1) Conditional feature extraction module**
  - **input 1) interpolated information $$\boldsymbol{X}_{o b}^0$$ 
  - input 2) adjacency matrix $$\boldsymbol{W}$$
  - output) global context with both spatial and temporal information as the condition for diffusion
- **(2) Noise estimation module**
  - utilizes this global context to **estimate the injected noises**
  - with a specialized attention mechanism to capture temporal dependencies and geographic information

<br>

Limitation: only works for the imputation of STGs with a single feature

<br>

# 5. TS Generation

TimeGAN 

- Proposed to generate TS data based on an integration of RNN and GAN
- GAN-based generative methods have been criticized as they are unstable & mode collapse

<br>

TimeVAE

- Requires a user-defined distribution for its probabilistic process

<br>

Probabilistic TS generator originated from diffusion models

- More flexible 
- Focus on TSGM (Lim et al., 2023) 
  - first and only work on this novel design. 

<br>

## (1) Problem Formulation

- MTS: $$\boldsymbol{X}^0=\left\{\boldsymbol{x}_1^0, \boldsymbol{x}_2^0, \ldots, \boldsymbol{x}_T^0 \mid \boldsymbol{x}_i^0 \in \mathbb{R}^D\right\}$$, 
- Goal: synthesize $$\boldsymbol{x}_{1: T}^0$$ 
  - by generating observation $$\boldsymbol{x}_t^0$$ at time point $$t \in[2, T]$$ with the consideration of its previous historical data $$x_{1: t-1}^0$$. 
  - Correspondingly, the target distribution is the conditional density $$q\left(\boldsymbol{x}_t^0 \mid \boldsymbol{x}_{1: t-1}^0\right)$$ for $$t \in[2, T]$$, and the associated generative process involves the recursive sampling of $$x_t$$ for all time points in the observed period. Details about the training and generation processes will be discussed in the next subsection.



## (2) TSGM

### Conditional score-based time series generative model (TSGM)

( Only work to study the TS generation based on the diffusion )

- Conditionally generate each TS observation based on the past generated observations
- Includes three components
  - (1) Encoder
  - (2) Decoder
  - (3) Conditional score-matching network
    - used to sample the hidden states, which are then converted to the TS samples via the decoder

- Input) MTS $$\boldsymbol{X}^0=\left\{\boldsymbol{x}_1^0, \boldsymbol{x}_2^0, \ldots, \boldsymbol{x}_T^0 \mid \boldsymbol{x}_i^0 \in \mathbb{R}^D\right\}$$
- Mapping) $$\boldsymbol{h}_t^0=\mathbf{E n}\left(\boldsymbol{h}_{t-1}^0, \boldsymbol{x}_t^0\right), \quad \hat{\boldsymbol{x}}_t^0=\mathbf{D e}\left(\boldsymbol{h}_t^0\right)$$
  - $$\hat{\boldsymbol{x}}_t^0$$ : Reconstructed TS at $$t$$ step
  - Recursive process 
    - En & De : constructed with the RNN
- Objective function (for both En & De) $$\mathcal{L}_{E D}$$ 
  - $$\mathcal{L}_{E D}=\mathbb{E}_{\boldsymbol{x}_{1: T}^0}\left[ \mid \mid \hat{\boldsymbol{x}}_{1: T}^0-\boldsymbol{x}_{1: T}^0 \mid \mid _2^2\right] $$.

<br>

Conditional score matching network $$s_{\boldsymbol{\theta}}$$

- Designed based on the SDE formulation of diffusion models
- Based on U-net
- Focuses on the **generation of hidden states** rather than producing the TS directly

<br>

Generation of hidden states

- Instead of applying the diffusion process to $$\boldsymbol{x}_t^0$$ ....

  Hidden states $$\boldsymbol{h}_t^0$$ is diffused to a Gaussian distribution by the following forward SDE

  - $$\mathrm{d} \boldsymbol{h}_t=f\left(k, \boldsymbol{h}_t\right) \mathrm{d} k+g(k) \mathrm{d} \boldsymbol{\omega}$$.
  - where $$k \in[0, K]$$ refers to the integral time. 

<br>

With the diffused sample $$\boldsymbol{h}_{1: t}^k$$,  $$s_{\boldsymbol{\theta}}$$ learns the gradient of the conditional log-likelihood function

Loss function: $$\mathcal{L}_{\text {Score }}=\mathbb{E}_{\mathbf{h}_{1: \mathbf{T}}^{\mathbf{0}}, \mathbf{k}} \sum_{t=1}^T[\mathcal{L}(t, k)]$$

- $$\mathcal{L}(t, k)=\mathbb{E}_{\boldsymbol{h}_t^k}\left[\delta(k) \mid \mid s_{\boldsymbol{\theta}}\left(\boldsymbol{h}_t^k, \boldsymbol{h}_{t-1}, k\right)-\nabla_{\boldsymbol{h}_t} \log q_{0 k}\left(\boldsymbol{h}_t \mid \boldsymbol{h}_t^0\right) \mid \mid ^2\right] $$.

<br>

### Training

(1) En & De

- pre-trained using the objective $$\mathcal{L}_{E D}$$. 

- ( can also be trained simultaneously with the network $$s_{\boldsymbol{\theta}}$$, but Lim et al. (2023) showed that the pre-training generally led to better performance )

<br>

(2) Score-matching network

- Hidden states are firstly obtained through inputting the entire TS $$\boldsymbol{x}_{1: T}^0$$ into the encoder
- Objective function $$\mathcal{L}_{\text {Score }}$$. 

<br>

### Sampling

- achieved by sampling hidden states & applying the decoder
  - analogous to solving the solutions to the time-reverse SDE.

- SOTA sampling quality and diversity

<br>

Limitation: more computationally expensive than GANs.

