---
title: ScoreGrad; Multivariate Probabilistic Time Series Forecasting with Continuous Energy-based Generative Models
categories: [TS, GAN]
tags: []
excerpt: 2021
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# ScoreGrad: Multivariate Probabilistic Time Series Forecasting with Continuous Energy-based Generative Models

<br>

![figure2](/assets/img/ts/img497.png)

# Contents

0. Abstract

0. 




<br>


# Abstract

Generative models in TS

- Many existing works can not be widely used because of the **constraints of functional form of generative models** or the **sensitivity to hyperparameters**

<br>

ScoreGrad

- MTS probabilistic forecasting
- Continuous energy-based generative models
- Composed of ..
  - (1) TS feature extraction module
  - (2) Conditional stochastic differential equation based score matching module
- Prediction: by iteratively solving reverse-time SDE
- First continuous energy based generative model used for TS forecasting

<br>

# 1. Introuction

Limitations of previous works

- [3], [4] : can not model stochastic information in TS
- [5], [6] : can not model long range time dependencies

<br>

Generative models

- proved effective for sequential modeling
- WaveNet [7] : proposes TCN with dilation
- [8] : Combines transformer + masked autoregressive flow together
  - however, functional form the fothe models based on VAE & flow based models are constrained
- TimeGrad [9] : uses an energy-based generative model (EBM) which transforms data distribution to target distribution by slowly injecting noise



### TimeGrad

- less restrictive on functional forms compared with VAE and flow based models
- still has some limitations
  - (1) DDPM in TimeGrad is sensitive to the noise scales
  - (2) Number of steps used for noise injection needs to be carefully designed
  - (3) Sampling methods for generation process in DDPM can be further extended

- solution: ScoreGrad

<br>

### ScoreGrad

- General framework for MTS forecasting based on continuous energy-based generative models

- DDPM = discrete form of a stochastic differential equation (SDE)

  $\rightarrow$ number of steps can be replaced by the interval of integration

  $\rightarrow$ noise scales can be easily tuned by diffusion term in SDE

  $\rightarrow$ ***conditional continuous energy based generative model + sequential modeling*** for forecasting!

- prediction : by iteratively sampling from the reverse continuous-time SDE

  ( any numerical solvers for SDEs can be used for sampling )

<br>

### Contribution

1. First continuous energy-based generative model used for MTS forecasting
2. General framework based on continuous energy-based generative models for TS forecasting
   - composed of a (1) time series feature extraction module and a (2) conditional SDE based score matching module
   - prediction : by solving reverse time SDE

3. Acheive SOTA

<br>

# 2. Related Work

## (1) MTS forecasting

pass

<br>

## (2) Energy-based generative models

Energy-based models (EBMs)

- Un-normalized probabilistic models

- Compared with VAE & flow-based generative models...

  - EBMs directly estimate the unnormalized negative log-probability
  - do not place a restriction on the tractability of the normalizing constants

  $\rightarrow$ much less restrictive in functional form

- limitations: the unknown normalizing constant of EBMs will make training particularly difficult

<br>

Several methods for training EBMS

- (1) Maximum likelihood estimation based on MCMC
  - [26] : estimate the gradient of the log-likelihood with MCMC sampling methods, instead of directly computing the likelihood. 
- (2) Score matching based methods. 
  - [28] : minimize a discrepancy between the gradient of the loglikelihood of data distribution and estimated distribution with Fisher divergence
    - optimization is computationally expensive! 
- (3) Noise contrastive estimation.
  - [29] : EBM can be learned by contrasting it with known density. 

<br>

# 3. Score-based generative models

## (1) Score matching models

$\mathbf{x} \sim p_{\mathcal{X}}(\mathbf{x})$ : the distribution of a $\mathrm{D}$ dim dataset

<br>

Score matching

- EBM which is proposed for learning nonnormalized statistical models
- aims to minimize the distance of the derivatives of the log-density function between data and model distributions.

<br>

 Although the density function of data distribution is unknown, the objective can be simplified as Eq. 1 based on a simple trick of partial integration.
$$
\begin{aligned}
& L(\theta)=\frac{1}{2} \mathbb{E}_{p_{\mathcal{X}}(\mathbf{x})}\left\|\nabla_{\mathbf{x}} \log p_\theta(\mathbf{x})-\nabla_{\mathbf{x}} \log p_{\mathcal{X}}(\mathbf{x})\right\|_2^2 \\
& \quad=\mathbb{E}_{p_{\mathcal{X}}(\mathbf{x})}\left[\operatorname{tr}\left(\nabla_{\mathbf{x}}^2 \log p_\theta(\mathbf{x})\right)+\frac{1}{2}\left\|\nabla_{\mathbf{x}} \log p_\theta(\mathbf{x})\right\|_2^2\right]+\mathrm{const}
\end{aligned}
$$
where $p_\theta(\mathbf{x})$ represents the distribution of model estimated by neural network and $\theta$ are learnable parameters of the model. $\nabla_{\mathbf{x}} \log p_\theta(\mathbf{x})$ is called score function. It's obvious that the optimal solution to Eq. 1 equals to $\nabla_{\mathbf{x}} \log p_{\mathcal{X}}(\mathbf{x})$ for all $\mathbf{x}$ and $t$.

## (2) Discrete score matching models

