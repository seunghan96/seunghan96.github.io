---
title: (paper 22) DeepCluster
categories: [CL, CV]
tags: []
excerpt: 2019
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Deep Clustering for Unsupervised Learning of Visual Features

<br>

## Contents

0. Abstract
0. Method
   0. Preliminaries
   0. Unsupervised Learning by Clustering
   0. Avoiding Trivial Solutions


<br>

# 0. Abstract

### DeepCluster

- end-to-end training of visual features

- jointly learns….
  - (1) the parameters of NN
  - (2) cluster assignments of the resulting features
- iteratively …
  - step 1) **groups** the features with standard clustering algorithm (k-means)
  - step 2) use the **subsequent assignments** as supervision to update NN

<br>

# 1. Method

![figure2](/assets/img/cl/img50.png)

<br>

## (1) Preliminaries

data : $$X=\left\{x_{1}, x_{2}, \ldots, x_{N}\right\}$$

model :

- (1) convent mapping : $$f_{\theta}$$
- (2) classifier : $$g_{W}$$

<br>

Loss function : $$\min _{\theta, W} \frac{1}{N} \sum_{n=1}^{N} \ell\left(g_{W}\left(f_{\theta}\left(x_{n}\right)\right), y_{n}\right)$$.

( $$l$$ : negative log-softmax )

<br>

## (2) Unsupervised Learning by Clustering

cluster the **output of the convnet** & 

use the subsequent cluster assignments as **“pseudo-labels”** to optimize $$\min _{\theta, W} \frac{1}{N} \sum_{n=1}^{N} \ell\left(g_{W}\left(f_{\theta}\left(x_{n}\right)\right), y_{n}\right)$$

( iteratively **learns the features** & **groups them** )

<br>

### K-means

- input : $$f_{\theta}\left(x_{n}\right)$$
- output : clusters them into $$k$$ distinct groups

- jointly learns a ..
  - (1) $$d \times k$$ centroid matrix $$C$$
  - (2) cluster assignments $$y_{n}$$ of each image $$n$$
- loss function :
  - $$\min _{C \in \mathbb{R}^{d \times k}} \frac{1}{N} \sum_{n=1}^{N} \min _{y_{n} \in\{0,1\}^{k}} \mid \mid f_{\theta}\left(x_{n}\right)-C y_{n} \mid \mid _{2}^{2} \quad \text { such that } y_{n}^{\top} 1_{k}=1 $$.

$$\rightarrow$$ these assignments are used as ***pseudo-labels***

( do not use **centroid matrix $$C$$** )

<br>

problem : **degenerating problem**

$$\rightarrow$$ cluster them into a single group… solution??

<br>

## (3) Avoiding Trivial Solutions

solutions are typically based on **constraining or penalizing the MINIMAL number of points per cluster**

but…not applicable to **large scale datasets**

<br>

### a) empty clusters

when cluster becomes empty….

randomly select a **non-empty cluster** & use **its centroid** with a **small perturbation** as a new centroid

<br>

### b) trivial parameterization

minimizing $$\min _{\theta, W} \frac{1}{N} \sum_{n=1}^{N} \ell\left(g_{W}\left(f_{\theta}\left(x_{n}\right)\right), y_{n}\right)$$

$$\rightarrow$$ leads to a **trivial parameterization**

( = predict the same output, regardless of output )

<br>

solution ?

- sample images based on a **uniform distn over the classes** ( or pseudo labels )

  ( = weighted loss, with weight of **inverse of size of clusters** )

<br>
