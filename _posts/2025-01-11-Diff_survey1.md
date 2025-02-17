---
title: (Diffusion survey) (Part 1; xxx)
categories: [MULT, LLM, NLP, CV, DIFF]
tags: []
excerpt: Diffusion Models and Representation Learning; A Survey (TPAMI 2024)
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Diffusion Models and Representation Learning: A Survey

https://arxiv.org/pdf/2304.00685

<br>

# Contents

- 

<br>

# Abstract

**Diffusion Models**

- Popular **generative modeling** methods 
- Unique instance of **SSL** methods (due to their independence from label annotation)

<br>

This paper: 

- Explores the interplay between **(1) diffusion models** and **(2) representation learning**
- Overview of diffusion models’ essential aspects, including ..
  - (1) **Mathematical foundations**
  - (2) **Popular denoising network architectures**
  - (3) **Guidance methods**
- Frameworks that leverage r**epresentations learned from pre-trained diffusion models** for subsequent recognition tasks 
- Methods that utilize **advancements in SSL** to enhance **diffusion models**
- Comprehensive overview of the taxonomy between diffusion models and representation learning

<br>

# 1. Introduction

### P1) Intro to diffusion models

Recently emerged as the SOTA of generative modeling

<br>

### P2) SSL

Scalability

- Current SOTA SSL show great scalability!
- Diffusion models exhibit similar scaling properties

<br>

Generation

- **Controlled generation approaches**

  - e.g., Classifier Guidance [43] and Classifier-free Guidance [67] 
    - Rely on annotated data $\rightarrow$ Bottleneck for scaling up!

- ***Guidance approaches that leverage "representation learning"*** 

  $\rightarrow$ Potentially enabling diffusion models to train on much **larger, annotation-free** datasets.

<br>

### P3) Diffusion & representation learning 

Two central perspectives

- (1) Using diffusion models **themselves** for representation learning 
- (2) Using representation learning for **improving** diffusion models. 

<br>

### P4) Increasing works

![figure2](/assets/img/llm/img538.png)

<br>

### P5) 

Current approaches:

$\rightarrow$ Rely on using diffusion models ***solely trained for generative synthesis*** for representation learning. 

<br>

Qualitative results 

![figure2](/assets/img/llm/img538.png)

<br>

### P6) Main contributions

- **(1) Comprehensive Overview**
  - Interplay between diffusion models and representation learning
  - How diffusion models can be used for representation learning and vice versa
- **(2) Taxonomy of Approaches**
  - Approaches in diffusion-based representation learning
- **(3) Generalized Frameworks**
  - Generalized frameworks for both ...
    - (1) diffusion model feature extraction 
    - (2) assignment-based guidance
- **(4) Future Directions**

<br>

# 2. Background

The following section outlines the required mathematical foundations of diffusion models. We also highlight current architecture backbones of diffusion models and provide a brief overview of sampling methods and conditional generation approaches.

<br>

## (1) Mathematical Foundations

### a) Forward process

$\begin{gathered}
p\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \quad \beta_t \mathbf{I}\right), \\
\forall t \in\{1, \ldots, T\}
\end{gathered}$.

<br>

$p\left(\mathbf{x}_t \mid \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{\bar{\alpha}_t} \mathbf{x}_0 ;\left(1-\bar{\alpha}_t\right) \mathbf{I}\right)$.

- where $\alpha_t:=1-\beta_t$ and $\bar{\alpha}_t:=\prod_{i=1}^t \alpha_i$. 

$\mathbf{x}_t=\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{\left(1-\bar{\alpha}_t\right)} \epsilon_t$.

<br>

### b) Backward process

$\mathbf{x}_T \sim \pi\left(\mathbf{x}_T\right)=\mathcal{N}(0, \mathbf{I})$ .

$p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)= \mathcal{N}\left(\mathbf{x}_{t-1} ; \mu_\theta\left(\mathbf{x}_t, t\right), \Sigma_\theta\left(\mathbf{x}_t, t\right)\right)$. 

<br>

### c) Loss function

$\begin{aligned}
\mathcal{L}_{v t b}= & -\log p_\theta\left(\mathbf{x}_0 \mid \mathbf{x}_1\right)+D_{K L}\left(p\left(\mathbf{x}_T \mid \mathbf{x}_0\right) \| \pi\left(\mathbf{x}_T\right)\right) \\
& +\sum_{t>1} D_{K L}\left(p\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0\right) \| p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)\right)
\end{aligned}$.

<br>

### d) Mean & Noise prediction

$\mu\left(\mathbf{x}_t, t\right):=\frac{\sqrt{\alpha_{t-1}}\left(1-\bar{\alpha}_{t-1}\right) \mathbf{x}_t+\sqrt{\bar{\alpha}_{t-1}}\left(1-\alpha_t\right) \mathbf{x}_0}{1-\bar{\alpha}_t}$.

$\mu_\theta\left(\mathbf{x}_t, t\right)=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right) .\right)$.

- DDPM: 
  - Suggest fixing the covariance $\Sigma_\theta\left(\mathbf{x}_t, t\right)$ to a constant value
  - Suggest predicting the added noise $\boldsymbol{\epsilon}\left(\mathbf{x}_t, t\right)$ instead of $\mathbf{x}_0$ 

- Loss function becomes...
  - $\mathcal{L}_{\text {simple }}=\mathbb{E}_{t \sim[1, T]} \mathbb{E}_{\mathbf{x}_0 \sim p\left(\mathbf{x}_0\right)} \mathbb{E}_{\boldsymbol{\epsilon}_{\mathrm{t}} \sim \mathcal{N}(0, \mathbf{I})}\left\|\boldsymbol{\epsilon}_{\mathrm{t}}-\boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t\right)\right\|^2 $.

<br>

### e) Improving sampling efficiency

**Velocity prediction** 

- Velocity = Linear combination of the denoised input & the added noise
- $\mathbf{v}=\bar{\alpha}_t \epsilon-\left(1-\bar{\alpha}_t\right) \mathbf{x}_t$.

$\rightarrow$ Combines benefits of both **data and noise parametrizations**

<br>

### f) Stochastic Differential Equation (SDE)

***Continuous (O) Discrete (X) timeseteps***

Diffusion process = **Continuous** time-dependent function $\sigma(t)$. 

<br>

$d \mathbf{x}=\mathbf{f}(\mathbf{x}, t) d t+g(t) d \mathbf{w}$.

- Vector-valued drift coefficient $\mathbf{f}(\cdot, t): \mathbb{R}^d \rightarrow \mathbb{R}^d$ 
- Scalar-valued diffusion coefficient $g(\cdot): \mathbb{R} \rightarrow \mathbb{R}$ 
- $\mathbf{w}$: standard Wiener process

<br>

Two widely used choices of the SDE formulation

- (1) Variance-Preserving (VP) SDE
- (2) Variance-Exploding (VE) SDE

<br>

(1) Variance-Preserving (VP) SDE, used in the work of Ho et al. [68] which is given by $\mathbf{f}(\mathbf{x}, t)=-\frac{1}{2} \beta(t) \mathbf{x}$ and $g(t)=\sqrt{\beta(t)}$, where $\beta(t)=\beta_t$ as $T$ goes to infinity. Note that this is equivalent to the continuous formulation of the DDPM parametrization in Equation 1. 



The second is the Variance-Exploding (VE) SDE [153] resulting from a choice of $\mathbf{f}(\mathbf{x}, t)=0$ and $g(t)=\sqrt{2 \sigma(t) \frac{d \sigma(t)}{d t}}$. The VE SDE gets its name since the variance continually increases with increasing $t$, whereas the variance in the VP SDE is bounded [154]. Anderson [7] derives an SDE that reverses a diffusion process, which results in the following when applied to the Variance Exploding SDE:
$$
d \mathbf{x}=-2 \sigma(t) \frac{d \sigma(t)}{d t} \nabla_{\mathbf{x}} \log p(\mathbf{x} ; \sigma(t)) d t+\sqrt{2 \sigma(t) \frac{d \sigma(t)}{d t}} d \mathbf{w}
$$

$\nabla_{\mathrm{x}} \log p(\mathrm{x} ; \sigma(t))$ is known as the score function. This score function is generally not known, so it needs to be approximated using a neural network. A neural network $D(\mathbf{x} ; \sigma)$ that minimizes the L2-denoising error can be used to extract the score function since $\nabla_{\mathbf{x}} \log p(\mathbf{x} ; \sigma(t))=$ $\frac{D(\mathrm{x} ; \sigma)-\mathrm{x}}{\sigma^2}$. This idea is known as Denoising Score Matching [161].
