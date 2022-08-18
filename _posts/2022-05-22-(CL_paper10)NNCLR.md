---
title: (paper 10) With a Little Help from My Friends; Nearest-Neighbor Contrastive Learning of Visual Representations
categories: [CL]
tags: []
excerpt: 2021
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# With a Little Help from My Friends: Nearest-Neighbor Contrastive Learning of Visual Representations

<br>

## Contents

0. Abstract
1. Related Work
2. Approach
   1. Contrastive Instance Discrimination
   2. Nearest-Neighbor CLR (NNCLR)
   3. Implementation Details
   

<br>

# 0. Abstract

Use positive samples from **OTHER** instances in the dataset

propose **NNCLR**

- samples the **nearest neighbors** from the dataset in the latent space & treat them as **positives**

![figure2](/assets/img/cl/img32.png)

<br>

# 1. Related Work

## Queues and Memory Banks

use a **SUPPORT set** as memory during training

( similar to MoCO )

<br>

MoCo vs NNCLR

- [MoCo] uses elements of the queue as **NEGATIVES**
- [NNCLR] ~ **POSITIVES**

<br>

# 2. Approach

describe contrastive learning in the context of 

- (1) instance discrimination
- (2) SimCLR

<br>

## (1) Contrastive Instance Discrimination

### InfoNCE

$$\mathcal{L}_{i}^{\text {InfoNCE }}=-\log \frac{\exp \left(z_{i} \cdot z_{i}^{+} / \tau\right)}{\exp \left(z_{i} \cdot z_{i}^{+} / \tau\right)+\sum_{z^{-} \in \mathcal{N}_{i}} \exp \left(z_{i} \cdot z^{-} / \tau\right)}$$.

<br>

### SimCLR

uses 2 views of same image as **positive pair**

<br>

Given **mini-batch of images** $$\left\{x_{1}, x_{2} \ldots, x_{n}\right\}$$,

$$\rightarrow$$ 2 different augmentations (views) are generated

- (1) $$z_{i}=\phi\left(\operatorname{aug}\left(x_{i}\right)\right)$$
- (2) $$z_{i}^{+}=\phi\left(\operatorname{aug}\left(x_{i}\right)\right)$$

<br>

InfoNCE loss used in SimCLR : $$\mathcal{L}_{i}^{\text {SimCLR }}=-\log \frac{\exp \left(z_{i} \cdot z_{i}^{+} / \tau\right)}{\sum_{k=1}^{n} \exp \left(z_{i} \cdot z_{k}^{+} / \tau\right)}$$

- each embedding is $$l_{2}$$ normalized before the dot product is computed in the loss

<br>

## (2) Nearest-Neighbor CLR (NNCLR)

propose using $$z_{i}$$ 's NN in the support set $$Q$$ to form the **positive pair**

- $$\mathcal{L}_{i}^{\mathrm{NNCLR}}=-\log \frac{\exp \left(\mathrm{NN}\left(z_{i}, Q\right) \cdot z_{i}^{+} / \tau\right)}{\sum_{k=1}^{n} \exp \left(\mathrm{NN}\left(z_{i}, Q\right) \cdot z_{k}^{+} / \tau\right)}$$.
  - where $$\mathbf{N N}(z, Q)=\underset{q \in Q}{\arg \min } \mid \mid z-q \mid \mid _{2}$$

<br>

## (3) Implementation Details

(1) symmetric loss

- add $$-\log \left(\exp \left(\mathrm{NN}\left(z_{i}, Q\right) \cdot z_{i}^{+} / \tau\right) / \sum_{k=1}^{n} \exp \left(\mathrm{NN}\left(z_{k}, Q\right) \cdot z_{i}^{+} / \tau\right)\right.$$

<br>

(2) insipred by BYOL …

- pass $$z_{i}^{+}$$through a prediction head $$g$$ to produce embeddings $$p_{i}^{+}=g\left(z_{i}^{+}\right)$$. 
- then use $$p_{i}^{+}$$instead of $$z_{i}^{+}$$in $$\mathcal{L}_{i}^{\mathrm{NNCLR}}=-\log \frac{\exp \left(\mathrm{NN}\left(z_{i}, Q\right) \cdot z_{i}^{+} / \tau\right)}{\sum_{k=1}^{n} \exp \left(\mathrm{NN}\left(z_{i}, Q\right) \cdot z_{k}^{+} / \tau\right)}$$.

<br>

### Support Set

implement support set as **queue**

- dimension : $$[m, d]$$
  - $$m$$ : size of queue
  - $$d$$ : size of embeddings
