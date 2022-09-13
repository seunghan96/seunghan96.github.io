---
title: (paper) SSL01 - Entropy Minimization
categories: [SSL]
tags: []
excerpt: 2004
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Semi-Supervised Learning by Entropy Minimization (2004)

<br>

## Contents

0. Abstract
1. Introduction
2. Derivation of the Criterion
   1. Likelihood
   2. When are Unlabeled Examples Informative?
   3. Entropy Regularization

<br>

# 0. Abstract

Semi-supervised Learning

- motivate **minimum entropy regularization**

<br>

# 1. Introduction

Supervised learning

- dataset : $$\mathcal{L}_n=\left\{\mathbf{x}_i, y_i\right\}_{i=1}^n$$
  - where $$\mathbf{x}_i \in \mathcal{X}$$ & $$y_i \in \Omega=\left\{\omega_1, \ldots, \omega_K\right\}$$

Semi-supervised learning

- $$y_i \in \Omega=\left\{\omega_1, \ldots, \omega_K\right\}$$ are limited to a subset of $$\mathcal{L}_n$$.

<br>

Semi-superviesd learning as **missing data problem**

$$\rightarrow$$ can be addressed by **generative models**

- generative models : $$P(X, Y)$$

  - more demanding than discriminative models $$P(Y \mid X)$$

  - more parameters are needed & resulting in more uncertainty

<br>

Propose an estimation principle applicable to any probabilistic classifier

<br>

# 2. Derivation of the Criterion

## (1) Likelihood

dataset : $$\mathcal{L}_n=\left\{\mathbf{x}_i, \mathbf{z}_i\right\}_{i=1}^n$$

- $$\mathbf{z} \in\{0,1\}^K$$ : dummy variable representing the actually available labels ( K classes )
  - if $$\mathbf{x}_i$$ is labeled $$\omega_k$$ : 
    - $$\mathbf{z}_{i k}=1$$
    - $$\mathbf{z}_{i \ell}=0$$ for $$\ell \neq k$$
  - if $$\mathbf{x}_i$$ is unlabeled :
    - $$\mathbf{z}_{i \ell}=1$$ for all $$l$$

<br>

Assumption : MAR (Missing At Random)

- for all unlabeled examples,  $$P\left(\mathbf{z} \mid \mathbf{x}, \omega_k\right)=P\left(\mathbf{z} \mid \mathbf{x}, \omega_{\ell}\right)$$, for any $$\left(\omega_k, \omega_{\ell}\right)$$ pair

- it implies that $$P\left(\omega_k \mid \mathbf{x}, \mathbf{z}\right)=\frac{z_k P\left(\omega_k \mid \mathbf{x}\right)}{\sum_{\ell=1}^K z_{\ell} P\left(\omega_{\ell} \mid \mathbf{x}\right)} .$$

<br>

( under iid ) conditional log-likelihood of $$(Z \mid X)$$ :

- $$L\left(\boldsymbol{\theta} ; \mathcal{L}_n\right)=\sum_{i=1}^n \log \left(\sum_{k=1}^K z_{i k} f_k\left(\mathbf{x}_i ; \boldsymbol{\theta}\right)\right)+h\left(\mathbf{z}_i\right)$$,
  - $$h(\mathbf{z})$$ : does not depend on $$P(X, Y)$$
  - $$f_k(\mathbf{x}_i ; \theta)$$ : model of $$P(\omega_k \mid \mathbf{x})$$

<br>

Goal : maximize $$L\left(\boldsymbol{\theta} ; \mathcal{L}_n\right)$$

<br>

## (2) When are Unlabeled Examples Informative?

*Information content of unlabeled examples decreases as classes overlap!*

<br>

Conditional entropy $$H(Y \mid X)$$ :

- measure of class overlap
- Invariant to the parameterization of the model
- related to the ***usefulness of unlabeled data***

<br>

Measure the **conditional entropy of class labels**, conditioned on the **observed variables**

- $$H(Y \mid X, Z)=-E_{X Y Z}[\log P(Y \mid X, Z)]$$.

<br>

Prior :

- in Bayesian framework, **assumptions are encoded using “prior”**
  - assumption : expect **high conditional entropy** 

- ***Maximum Entropy Prior*** : $$P(\boldsymbol{\theta}, \boldsymbol{\psi}) \propto \exp (-\lambda H(Y \mid X, Z)))$$
  - $$(\boldsymbol{\theta}, \boldsymbol{\psi})$$ : parameters of model $$P(X,Y,Z)$$
  - $$E_{\Theta \Psi}[H(Y \mid X, Z)]=c$$, where
    - $$c$$ : quantifies **how small the entropy should be on average**

<br>

Empirical estimation of entropy

- $$H_{\mathrm{emp}}\left(Y \mid X, Z ; \mathcal{L}_n\right)=-\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^K P\left(\omega_k \mid \mathbf{x}_i, \mathbf{z}_i\right) \log P\left(\omega_k \mid \mathbf{x}_i, \mathbf{z}_i\right)$$.

<br>

## (3) Entropy Regularization

Model of $$P(\omega_k \mid \mathbf{x}, \mathbf{z})$$ : $$g_k(\mathbf{x}, \mathbf{z} ; \boldsymbol{\theta})=\frac{z_k f_k(\mathbf{x} ; \boldsymbol{\theta})}{\sum_{\ell=1}^K z_{\ell} f_{\ell}(\mathbf{x} ; \boldsymbol{\theta})} $$

- (for labeled data) : $$g_k(\mathbf{x}, \mathbf{z} ; \boldsymbol{\theta})=z_k$$

- (for unlabeled data) : $$g_k(\mathbf{x}, \mathbf{z} ; \boldsymbol{\theta})=f_k(\mathbf{x} ; \boldsymbol{\theta})$$

<br>

**MAP estimate** : 

maximizer of posterior distn, which is

- $$\begin{aligned}
  C\left(\boldsymbol{\theta}, \lambda ; \mathcal{L}_n\right) &=L\left(\boldsymbol{\theta} ; \mathcal{L}_n\right)-\lambda H_{\mathrm{emp}}\left(Y \mid X, Z ; \mathcal{L}_n\right) \\
  &=\sum_{i=1}^n \log \left(\sum_{k=1}^K z_{i k} f_k\left(\mathbf{x}_i\right)\right)+\lambda \sum_{i=1}^n \sum_{k=1}^K g_k\left(\mathbf{x}_i, \mathbf{z}_i\right) \log g_k\left(\mathbf{x}_i, \mathbf{z}_i\right),(6)
  \end{aligned}$$.

  ( constant terms are dropped )

- (1) $$L\left(\boldsymbol{\theta} ; \mathcal{L}_n\right)$$ : only affected by **labeled data**

- (2) $$H_{\mathrm{emp}}\left(Y \mid X, Z ; \mathcal{L}_n\right)$$ : only affected by value of $$f_k(\mathbf{x})$$ On **unlabeled data**

 <br>
