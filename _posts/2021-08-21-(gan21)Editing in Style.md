---
title: \[Paper Review\] 21.(Analysis,Manipulation) Editing in Style
categories: [GAN]
tags: [GAN]
excerpt: 2020, local, semantically aware edits to output image
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# \[Paper Review\] 21. Editing in Style

<br>

### Contents

0. Abstract
1. Introduction
2. Method
   1. Characterizing units by Dissection
   2. Measuring Causal Relationships using Intervention

<br>

# 0. Abstract

ability to control & condition the output is still limited

$$\rightarrow$$ introduce a **simple & effective method** for making **local, semantically-aware edits** to a **target output image**

<br>

# 1. Related Works

goal is NOT to propose new GAN,

BUT to offer **local editing method** for its output

( by changing the style of specific objects )

<br.

## (1) GAN-based Image Editing

semantic image editing

- 1) latent code-based : for **GLOBAL** attribute editing
- 2) activation-based : for **LOCAL** ~

<br>

### Latent code-based 

- learn a manifold in latent space

- perform semantic edits, ***by traversing paths along this manifold***

- example )

  - use AE to disentangle image into **semantic subspaces** & reconstruct the image

  - **global changes** in color/light/;pose/...

<br>

### Activation-based

- directly manipulate **specific SPATIAL positions** on 

  activation tensor, at certain CNN layer

- example )

  - GAN Dissection controls the **presence/absence of objects** at given position

<br>

This paper focuses on ***latent code-based approach*** for **local editing**

- neither rely on external supervision
- nor involves complex spatial blending operations

<br>

# 2. Local Semantics in Generative Models

## (1) Feature Factorization

DFF (Deep Feature Factorization)

- explains CNN's learned representation, via **salicency maps**

- with this, it has been shown that...

  **CNNs learns features that act as (1) semantic object & (2) object-part detectors**

<br>

Inspired by DFF, conduct a similar analysis

- apply **spherical k-means** to $$C$$-dim **activation vectors**

  ( activation tensor : $$\mathbf{A} \in \mathbb{R}^{N \times C \times H \times W}$$ )

- clustering generates a tensor of **cluster membership**

  ( membership : $$\mathbf{U} \in\{0,1\}^{N \times K \times H \times W}$$ )

  - $$K$$ : user-defined

<br>

Result

- at certain layers of generator,

  cluster correspond well to semantic objects & parts

- each pixel in the heatmap is color-coded to indicate cluster index

<br>

![figure2](/assets/img/gan/img53.png)

<br>

### $$M_{k, c}$$ : Contribution of channel $$c$$ to semantic cluster $$k$$

- via cluster memberships, $$\mathbf{U} \in\{0,1\}^{N \times K \times H \times W}$$

- $$\boldsymbol{M}_{k, c}=\frac{1}{N \dot{H} \dot{W}} \sum_{n, h, w} \mathbf{A}_{n, c, h, w}^{2} \odot \mathbf{U}_{n, k, h, w}$$.

  - feature maps of $$\mathbf{A}_{l}$$ ~ N(0,1)

    $$\rightarrow$$ contribution : 0~1

<br>

## (2) Local Editing

### a) Style GAN review

- latent vector $$z$$ ~ prior

- $$z$$ is transformed to **intermediate latent vector** $$\boldsymbol{w} \in \mathbb{W}$$

  $$\rightarrow$$ show better **disentanglement properties*

- $$\mathbf{A} \in \mathbb{R}^{(C \times H \times W)}$$ : input to a convolutional layer

- $$w$$ : alters feature maps, via a **per-layer style**

- motivated by **style transfer**

<br>

### b) Conditioned Interpolation

Notation

- target image : $$S$$
- reference image : $$R$$

would like to transfer the appearance of a specified local object/part from $$R$$ to $$S$$

<br>

[ Global transfer ]

- $$\sigma^{G}=\sigma^{S}+\lambda\left(\sigma^{R}-\sigma^{S}\right)$$.

  where $$0 \leq \lambda \leq 1$$

<br>

[ Selective local editing ]

- control style interpolation with matrix transfomration

- $$\sigma^{G}=\sigma^{S}+Q\left(\sigma^{R}-\sigma^{S}\right)$$.

  - $$Q$$ : diagonal matrix ( where $$q \in[0,1]^{C}$$ )

    ( $$q$$ : query vector )

<br>

### c) Choosing the query

best query $$q$$ = one that favor channels that..

- affect the ROI (region of interest)
- while ignoring channels that have an effect outside the ROI

<br>

**[ Simple Approach ]**

- use $$M_{k^{\prime}, c}$$

- clipping $$\boldsymbol{q}_{c}=\min \left(1, \lambda \boldsymbol{M}_{k^{\prime}, c}\right)$$

  where $$\boldsymbol{q}_{c}$$ is the $$c$$-th channel element of $$\boldsymbol{q}$$,

- updates all channels at same time

<br>

**[ Proposed Approach ]**

- **sequential approach**

- first set the most relevant channel to the maximum slope of 1,

  before raising the slope of the second-most relevant, ...

- solve this by sorting channels based on $$M_{k^{\prime}}$$

  & greedily assigning $$q_c=1$$ to most relevant channels

$$\begin{gathered}
\underset{\boldsymbol{q}_{c}}{\arg \min }  \boldsymbol{q}_{c}\left[\boldsymbol{M}_{k^{\prime}, c}-\rho\left(1-\boldsymbol{M}_{k^{\prime}, c}\right)\right] \\
\quad \quad \quad \text { s.t. } \sum_{c=1}^{C} \boldsymbol{q}_{c}\left(1-\boldsymbol{M}_{k^{\prime}, c}\right) \leq \epsilon , \quad 0 \leq \boldsymbol{q}_{c} \leq 1
\end{gathered}$$

