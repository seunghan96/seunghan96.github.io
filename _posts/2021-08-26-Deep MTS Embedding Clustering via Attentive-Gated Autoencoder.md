---
title: (paper) Deep MTS Embedding Clustering via Attentive-Gated Autoencoder
categories: [TS]
tags: [TS]
excerpt: 2020, Time Series Clustering
---

# Deep MTS Embedding Clustering via Attentive-Gated Autoencoder (2020)

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Contents

0. Abstract
1. Introduction
2. DeTSEC : Deep Time Series Embedding Clustering

<br>

# 0. Abstract

propose a DL-based framework for clustering **MTS**, with **varying length**

$$\rightarrow$$ propose **DeTSEC (Deep Time Series Embedding Clustering)**

<br>

# 1. Introduction

**DeTSEC (Deep Time Series Embedding Clustering)**

- different domains OK
- varying length OK

<br>

2 stages

- step 1) Recurrent autoencoder exploits **attention** & **gating mechanism** to produce a preliminary embedding representation
- step 2) **Clustering Refinement stage**
  - stretch the embedding manifold towards the corresponding cluters

<br>

# 2. DeTSEC : Deep Time Series Embedding Clustering

Notation

- $$X=\left\{X_{i}\right\}_{i=1}^{n}$$ : multivariate time-series

  - $$X_{i} \in X$$ : time-series 

    where $$X_{i j} \in R^{d}$$ = multidimensional vector of the time-series $$X_{i}$$ at timestamp $$j$$, with $$1 \leq j \leq T$$ 

  - $$d$$ : dimensionality of $$X_{i j}$$ 

  - $$T$$ : maximum time-series length

- $$X$$ can contain time-series with **DIFFERENT** length

<br>

Goal

- partition $$X$$ in a given number of clusters

<br>

2 stages

- stage 1) GRU based autoencoder

  - for each GRU unit, **attention** is applied,

    to combine the information **coming from different timestamps**

- stage 2) refine the representation, by taking into account a 2-fold task

  - 1) reconstruction
  - 2) another one devoted to stretch the embedding manifold towards clustering centroids

<br>
3 different compontents

- 1) encoder
- 2-1) backward decoder
- 2-2) forward decoder

<br>

![figure2](/assets/img/ts/img121.png)

<br>

## Loss Function

(1) autoencoder network

$$\begin{aligned}
L_{a e}=& \frac{1}{ \mid X \mid } \sum_{i=1}^{ \mid X \mid } \mid \mid X_{i}-\operatorname{dec}\left(\operatorname{enc}\left(X_{i}, \Theta_{1}\right), \Theta_{2}\right) \mid \mid _{2}^{2} \\
&+\frac{1}{ \mid X \mid } \sum_{i=1}^{ \mid X \mid } \mid \mid \operatorname{rev}\left(X_{i}\right)-\operatorname{dec}_{b a c k}\left(\operatorname{enc}\left(X_{i}, \Theta_{1}\right), \Theta_{3}\right) \mid \mid _{2}^{2}
\end{aligned}$$.

<br>

(2) regularizer term

$$\frac{1}{ \mid X \mid } \sum_{i=1}^{ \mid X \mid } \sum_{l=1}^{n C l u s t} \delta_{i l}  \mid \mid  \text { Centroids }_{l}-\operatorname{enc}\left(X_{i}, \Theta_{1}\right)  \mid \mid _{2}^{2}$$.

<br>

(3) Total loss : (1) + (2)