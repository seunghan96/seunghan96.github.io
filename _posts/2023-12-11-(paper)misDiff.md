---
title: MissDiff; Training Diffusion Models on Tabular Data with Missing Values
categories: [GAN,DIFF,TAB]
tags: []
excerpt: ICML 2023 Workshop
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# MissDiff: Training Diffusion Models on Tabular Data with Missing Values

<br>

# Contents

0. Abstract

0. Introduction

0. Method
   0. Problem Setup
   0. Preliminaries
   0. MissDiff




# 0. Abstract

Unified and principled **diffusion-based** framework for learning from **data with missing values**

<br>

**Findings / Proposals**

- (1) **“Impute-then-generate”** pipeline may lead to a **biased learning objective**
- (2) Propose to **mask** the regression loss of Denoising Score Matching
  - consistent in learning the score of data distributions
  - training objective serves as an upper bound for the negative likelihood in certain cases. 
- (3) Experiments on multiple tabular datasets

<br>

# 1. Introduction

Previous work 

- Learning from data with missing values and synthesizing complete data
  - using GAN or VAE

- Involve training additional networks & impose certain assumptions on the missing mechanisms

<br>

### MissDiff

- Generative model training from data ***with missing values***

- ***Theoretical justifications*** of MissDiff on recovering the oracle score function and upper bounding the negative likelihood on the data under mild assumptions on the missing mechanisms

- First work that learns a generative model from ***mixed-type data containing missing values***

  & used directly in the training process ***without prior imputations***

<br>

# 2. Method

## (1) Problem Setup

Notation

- Data: $$\mathbf{x}=\left(x_1, \ldots, x_d\right) \in \mathcal{X}$$ sampled from $$p_0(\mathbf{x})$$. 

- Each variable $$x_i, i \in[d]$$:

  - can be either categorical or continuous.

- Binary mask $$\mathbf{m}=$$ $$\left(m_1, \ldots, m_d\right) \in\{0,1\}^d$$ i

  - i.e., $$m_i=1$$ if $$x_i$$ is observed, and $$m_i=0$$ if $$x_i$$ is missing. 

- Observed data $$\mathbf{x}^{\text {obs }}=\mathbf{x} \odot \mathbf{m}+$$ na $$\odot(\mathbf{1}-\mathbf{m})$$

  - na: missing value

- $$n$$ complete (unobservable) training data points $$\mathbf{x}_1, \ldots, \mathbf{x}_n \stackrel{i i d}{\sim} p_0(\mathbf{x})$$ 

  & $$n$$ corresponding masks $$\mathbf{m}_1, \ldots, \mathbf{m}_n$$ generated from a specific missing data mechanism 

- Observed data values are $$S^{\mathrm{obs}}=\left\{\mathrm{x}_i^{\mathrm{obs}}\right\}_{i=1}^n$$ 
  - with $$\mathbf{x}_i^{\text {obs }}=\mathbf{x}_i \odot \mathbf{m}_i+$$ na $$\odot\left(\mathbf{1}-\mathbf{m}_i\right)$$.

<br>

Goal: train a generative model $$p_\phi$$, using the observed data $$S^{\text {obs }}$$, 

- Mainly consider the **score-based generative model** as $$p_\phi$$.

<br>

## (2) Preliminaries: Score-based Generative Model

Forward

- $$\mathrm{d} \mathbf{x}(t)=\mathbf{f}(\mathbf{x}(t), t) \mathrm{d} t+g(t) \mathrm{d} \mathbf{w}$$.

Backward

- $$\mathrm{d} \mathbf{x}(t)=\left[\mathbf{f}(\mathbf{x}(t), t)-g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] \mathrm{d} t+g(t) \mathrm{d} \overline{\mathbf{w}}$$.

<br>

## (3) MissDiff

###  Impute-then-generate

- step 1) Impute the observed data $$\mathrm{x}^{\text {obs }}$$ using an imputation model $$f_{\varphi}$$. 
- step 2) Generative model is trained on imputed data, i.e., $$\left(\mathbf{x}^{\text {obs }}, \mathbf{x}^{\text {miss }}:=f_{\varphi}\left(\mathbf{x}^{\text {obs }}\right)\right)$$ 

$$\rightarrow$$ Such pipeline is ***typically biased*** !!!

( With **single** imputation, the **conditional distribution** over the missing data is discarded, and the optimal single imputation can no longer capture the data variability )

<br>

### MissDiff

- Diffusion-based framework for learning on **missing data**
- Incorporates the **uncertainty of missing data** directly into the learning process.

<br>

The score model $$\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}(t), t)$$ is learned as solution to....

$$\begin{aligned}
& \boldsymbol{\theta}^*=\underset{\boldsymbol{\theta}}{\arg \min } J_{D S M}(\boldsymbol{\theta}) \\
& :=\frac{T}{2} \mathbb{E}_t\left\{\lambda(t) \mathbb{E}_{p\left(\mathbf{x}^{\mathrm{obs}}(0), \mathbf{m}\right)} \mathbb{E}_{p\left(\mathbf{x}^{\mathrm{obs}}(t) \mid \mathbf{x}^{\mathrm{obs}}(0)\right)}\right. \\
& \left. \mid \mid \left(\mathbf{s}_{\boldsymbol{\theta}}\left(\mathbf{x}^{\mathrm{obs}}(t), t\right)-\nabla_{\mathbf{x}^{\mathrm{obs}}(t)} \log p\left(\mathbf{x}^{\mathrm{obs}}(t) \mid \mathbf{x}^{\mathrm{obs}}(0)\right)\right) \odot \mathbf{m} \mid \mid _2^2\right\}
\end{aligned}$$.

- $$\lambda(t)$$ : positive weighting function
- $$\mathbf{m}=$$ $$\mathbb{1}\left\{\mathrm{x}^{\mathrm{obs}}(0)=\right.$$ na $$\}$$ : the missing entries in $$\mathrm{x}^{\mathrm{obs}}(0)$$ 
- $$p\left(\mathbf{x}^{\text {obs }}(t) \mid \mathbf{x}^{\text {obs }}(0)\right)=\mathcal{N}\left(\mathbf{x}^{\text {obs }}(t) ; \mathbf{x}^{\text {obs }}(0), \beta_t \mathbb{I}\right)$$ :  Gaussian transition kernel

<br>

To make $$p\left(\mathbf{x}^{\text {obs }}(t) \mid \mathbf{x}^{\text {obs }}(0)\right)$$ and $$\nabla_{\mathbf{x}^{\text {obs }}(t)} \log p\left(\mathbf{x}^{\text {obs }}(t) \mid \mathbf{x}^{\text {obs }}(0)\right)$$ well defined for the mixedtype data, 

- **Use 0** to replace na for **"continuous variables"**
- **New category** to represent na for **"discrete variables"**
  - One-hot embedding is applied

