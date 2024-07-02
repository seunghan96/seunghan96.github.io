---
title: DTMamba; Dual Twin Mamba for Time Series Forecasting
categories: [TS,MAMBA]
tags: []
excerpt: arxiv 2024
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# DTMamba : Dual Twin Mamba for Time Series Forecasting

<br>

# Contents

1. Introduction
2. Proposed Method
   1. Problem Statement
   2. Normalization
   3. CI & Reverse CI
   4. Twin Mamba
3. Experiments

<br>

# 1. Introduction

### DTMamba (Dual Twin Mamba)

[Procedure]

- RevIN
  - CI layer
    - TMamba blocks ($$\times 2$$)
    - Projection layer
  - revese CI
- reverse RevIN

<br>

# 2. Proposed Method

![figure2](/assets/img/ts2/img148.png)

Three layers

- (1) CI layer
- (2) TMamba block
  - embedding
  - FC layer
  - dropout
  - pair of twin Mambas
- (3) Projection layer

<br>

## (1) Problem Statement

**Multivariate TS**

 $$X=\left\{x_1, \ldots, x_L\right\}$$.

- $$X_i \in \mathbb{R}^N$$ consists of $$N$$ dimensions $$\left\{x_i^1, \ldots, x_i^N\right\}$$ 

<br>

**TS forecasting**

- $$X=\left\{x_1, \ldots, x_T\right\} \in$$ $$\mathbb{R}^{T \times N}$$,
- $$\hat{X}=\left\{\hat{x}_{T+1}, \ldots, \hat{x}_{T+S}\right\} \in \mathbb{R}^{S \times N}$$ ,

<br>

## (2) Normalization

 $$X^0=\left\{x_1^0, \ldots, x_T^0\right\} \in \mathbb{R}^{T \times N}$$, via $$X^0=\operatorname{RevIN}(X)$$.

<br>

## (3) CI & Reverse CI

(B,T,N) $$\rightarrow$$ (BxN, 1, T)

<br>

## (4) Twin Mamba

![figure2](/assets/img/ts2/img149.png)

<br>

### a) Embedding Layes

Embed the $$X^I$$ into $$X^E:(B \times$$ $$N, 1, n i)$$. 

<br>

DTMamba 

- consists of two TMamba Block in total

<br>

Embedding layer 

- Embed the TS into $$\mathrm{n} 1$$ and $$\mathrm{n} 2$$ dimension
  - Embedding 1 & Embedding 2

<br>

### b) Residual

- To prevent overfitting
- FC layer to change the dimension of the residual

<br>

### c) Dropout

$$X^E$$ $$\rightarrow$$ $$X^D:(B \times N, 1, n i)$$.

<br>

### d) MAMBAs

TMamba Block = Two parallel Mamba. 

- Multi-level feature learning can be achieved.

- Mamba (1)
  - learn low-level temporal features
- Mamba (2)
  - learn high-level temporal patterns

<br>

### e) Projection Layer

$$R^1$$ and $$R^2$$

- Representation learned by the two pairs of TMamba Block

<br>

Step 1) Addition operation

- $$X^A:(B \times N, 1, n 2) \leftarrow X^I+R^1+R^2$$.

Step 2) Prediction ( next length $$S$$ )

- Use a linear layer
- Get $$X^P:(B \times N, 1, S)$$,

Step 3) Reverse CI ( = reshape ) 

- $$\hat{X}:(B, S, N)$$.

<br>

# 3. Experiments

![figure2](/assets/img/ts2/img150.png)

![figure2](/assets/img/ts2/img151.png)

![figure2](/assets/img/ts2/img153.png)
