---
title: (paper 31) Invariance Propagation
categories: [CL, CV]
tags: []
excerpt: 2020
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Unsupervised Representation Learning by Invariance Propagation

<br>

## Contents

0. Abstract

0. BigBiGAN
   1. Encoder $$\mathcal{E}$$
   
   2. Joint Discriminator $$\mathcal{D}$$ 


<br>

# 0. Abstract

Unsupervised learning based on contrastive learning

- aim to learn representations, invariant to **instance-level variations**

<br>

propose **Invariance Propagation**

- focus on learning representations, invariant to **category-level variations**

- **recursively** discovers semantically consistent samples, which are in the **same high-density regions**
- **hard sampling 

combining (1) clustering + (2) representation learning

$$\rightarrow$$ doing it naively...leads to **degenerate solutions**

<br>

solution : propose a method, that **maximizes the information between labels & input data indicies**

<br>

# 1. Introduction

self-supervision tasks : mostly done by **new pretext task**

But, **task of classification is sufficient for pre-training**

( of course.... provided that ***labels are given*** )

$$\rightarrow$$ focus on **obtaining the labels automatically** ( with **self-labeling algorithm** )

<br>

Degeneration problem ?

$$\rightarrow$$ solve by **adding the constraint**, that the **labels must induce an equipartition of the data** ( = maximizes the information between data indicies & labels )

<br>

# 2. Method

(1) self-labeling method

(2) interpret the method as optimizing laels & targets of CE loss

<br>

## (1) Self-labeling

Notation :

- $$x=\Phi(I)$$ : DNN
  - map images ($$I$$) to feature vectors ($$x \in \mathbb{R}^D$$ )
- $$I_1, \ldots, I_N$$ : Image data
- $$y_1, \ldots, y_N \in\{1, \ldots, K\}$$ : Image labels
- $$h: \mathbb{R}^D \rightarrow \mathbb{R}^K$$ : classification head
- $$p\left(y=\cdot \mid \boldsymbol{x}_i\right)=\operatorname{softmax}\left(h \circ \Phi\left(\boldsymbol{x}_i\right)\right)$$ : class probabilities 

<br>

Train model & head parameters, with **average CE loss**

- $$E\left(p \mid y_1, \ldots, y_N\right)=-\frac{1}{N} \sum_{i=1}^N \log p\left(y_i \mid \boldsymbol{x}_i\right)$$.

$$\rightarrow$$ requires **labelled dataset**

( if not, requires a ***self-labeling mechanism*** )

<br>

[ Self-labeling mechanism ]

- achieved by **jointly optimizing** , w.r.t

  - (1) model $$h \circ \Phi$$
  - (2) labels $$y_1, \ldots, y_N$$

- but if fully unsupervised .... leads to **degenerate solution**

  ( = trivially minimized by assigning all data points to a single (arbitrary) label )

<br>

Solution?

- first, encode the labels as **posterior distn** $$q\left(y \mid \boldsymbol{x}_i\right)$$

  - (Before) $$E\left(p \mid y_1, \ldots, y_N\right)=-\frac{1}{N} \sum_{i=1}^N \log p\left(y_i \mid \boldsymbol{x}_i\right)$$.
  - (After) $$E(p, q)=-\frac{1}{N} \sum_{i=1}^N \sum_{y=1}^K q\left(y \mid \boldsymbol{x}_i\right) \log p\left(y \mid \boldsymbol{x}_i\right) .$$

  ( optimizing $$q$$ = reassigning labels )

- to avoid degeneracy...

  $$\rightarrow$$ add the constraint that **the label assignments must partition the data in equally-sized subsets**

- objective function :

  - $$\min _{p, q} E(p, q) \quad \text { subject to } \quad \forall y: q\left(y \mid \boldsymbol{x}_i\right) \in\{0,1\} \text { and } \sum_{i=1}^N q\left(y \mid \boldsymbol{x}_i\right)=\frac{N}{K}$$.

  