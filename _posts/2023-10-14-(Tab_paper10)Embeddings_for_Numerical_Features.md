---
title: On Embeddings for Numerical Features in Tabular Deep Learning
categories: [TAB]
tags: []
excerpt: NeurIPS 2022
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# On Embeddings for Numerical Features in Tabular Deep Learning (NeurIPS 2022)

https://openreview.net/pdf?id=pfI7u0eJAIr

<br>

# Contents

0. Abstract



<br>

# Abstract

MLP = map scalar values of numerical features to high-dim embeddings

However ... **embeddings for numerical features are underexplored**

<br>

Propose 2 different approaches to build embedding modules

- (1) Piecewise linear encoding
- (2) Periodic activations

$$\rightarrow$$ Beneficial for many backbones

<br>

# 1. Introduction

Most previous works :

- focus on developing **more powerful backbone** 
- overlook the **design of embedding modules**

<br>

This paper demonstrate that the **embedding step has a substantial impact on the model effectiveness**

Two different building blocks for constructing embeddings

- (1) Piecewise linear encoding
- (2) Periodic activation functions

<br>

# 2. Related Work

## (1) Tabular DL

Do not consistently outperform **GBDT models**

Do not consistently outperform **properly tuned simple models** ( MLP, ResNet )

<br>

## (2) Transformers in Tabular DL

Requires mapping the scalar values to high-dim vectors

Existing works: relatively simple computational blocks

- ex) FT-Transformer : use single linear layer

<br>

## (3) Feature binning

Discretization technique

( numerical features $$\rightarrow$$ categorical features )

<br>

## (4) Periodic activations

key component in processing **coordinate-like inputs**

<br>

# 3. Embeddings for numerical features

### Notation

Dataset : $$\left\{\left(x^j, y^j\right)\right\}_{j=1}^n$$ 

-  $$y^j \in \mathbb{Y}$$ represents the object's label
- $$x^j=\left(x^{j(n u m)}, x^{j(c a t)}\right) \in \mathbb{X}$$ .

<br>

## (1) General Framework

**"embeddings for numerical features"**

- $$z_i=f_i\left(\left(x_i^{(\text {num })}\right) \in \mathbb{R}^{d_i}\right.$$,
  -  where $$f_i(x)$$ is the embedding function for the $$i$$-th numerical feature
- all features are computed independently of each other.

<br>

## (2) Piecewise Linear encoding

![figure2](/assets/img/tab/img52.png)

<br>

$$\begin{aligned}
& \operatorname{PLE}(x)=\left[e_1, \ldots, e_T\right] \in \mathbb{R}^T \\
& e_t= \begin{cases}0, & x<b_{t-1} \text { AND } t>1 \\
1, & x \geq b_t \text { AND } t<T \\
\frac{x-b_{t-1}}{b_t-b_{t-1}}, & \text { otherwise }\end{cases}
\end{aligned}$$.

<br>

**Note on attention-based models**

Order-invariant ... need positional information ( = feature index information )

$$\rightarrow$$ place one linear layer after PLE ( = same effect as above )

$$f_i(x)=v_0+\sum_{t=1}^T e_t \cdot v_t=\operatorname{Linear}(\operatorname{PLE}(x))$$.

<br>

### a) Obtaining bins from quantiles

From empirical quantile

- $$b_t=\mathbf{Q}_{\frac{t}{T}}\left(\left\{x_i^{j(\text { num })}\right\}_{j \in J_{\text {train }}}\right)$$.

<br>

### b) Building target-aware bins

Supervised approach for constructing bins

- recusrively splits its value range in a greedy manner using target as guidance

  ( = like decision tree )

<br>

## (3) Periodic activation functions

Train the pre-activation coefficient ( instead of fixed )

$$f_i(x)=\operatorname{Periodic}(x)=\operatorname{concat}[\sin (v), \cos (v)], \quad v=\left[2 \pi c_1 x, \ldots, 2 \pi c_k x\right]$$.

- where $$c_i$$ are trainable parameters initialized from $$\mathcal{N}(0, \sigma)$$. 
  - $$\sigma$$ : important hyperparamter 
  - tune both $$\sigma$$ and $$k$$

<br>

# 4. Experiments

## (1) Datasets

![figure2](/assets/img/tab/img53.png)

<br>

## (2) Model Names

![figure2](/assets/img/tab/img54.png)

<br>

## (3) Results

![figure2](/assets/img/tab/img55.png)

![figure2](/assets/img/tab/img56.png)

<br>

## (4) DL vs. GBDT

![figure2](/assets/img/tab/img57.png)
