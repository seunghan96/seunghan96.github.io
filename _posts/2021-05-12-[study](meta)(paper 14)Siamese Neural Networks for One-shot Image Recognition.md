---
title: \[meta\] (paper 14) Siamese Neural Networks for One-shot Image Recognition
categories: [META,STUDY]
tags: [Meta Learning]
excerpt: 2015
---

# Siamese Neural Networks for One-shot Image Recognition (2015)

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Contents

0. Abstract
1. Approach
2. Deep Siamese Networks for Image Verification
   1. Model
   2. Learning

<br>

# 0. Abstract

Siamese NN : rank similarity between inputs

- not just to new data
- but also to **"new class"**

<br>

# 1. Approach

learn image representations, via supervised **"metric-based approach"**

with **"SIAMESE neural network"**

& reuse network's features for one-shot learning

<br>

Employ **"Siamese CNN"**

- 1) capable of learning generic **"image feature"**
- 2) easily trained using **standard optimization techniques**
- 3) does NOT rely upon **domain-specific knowledge**

<br>

Pairing with the HIGHEST score ( according to verification network )

$$\rightarrow$$ HIGHEST probability for one-shot task

<br>

# 2. Deep Siamese Networks for Image Verification

Siamese Net : 1990s에 처음 소개

- consists of twin networks

<br>

This paper

- use weighted $$L_1$$ distance between twin feature vectors $$\mathbf{h_1}$$ & $$\mathbf{h_2}$$
- sigmoid activation 사용
- cross-entropy objective

<br>

## 2-1. Model

![figure2](/assets/img/META/img39.png)

<br>

![figure2](/assets/img/META/img38.png)

<br>

## 2-2. Learning

### (1) Loss Function

Notation

- $$M$$ : minibatch size 
- $$\mathbf{y}\left(x_{1}^{(i)}, x_{2}^{(i)}\right)$$ : length- $$M$$ vector which contains the labels for the minibatch
  - $$y\left(x_{1}^{(i)}, x_{2}^{(i)}\right)=1$$ : same class
  - $$y\left(x_{1}^{(i)}, x_{2}^{(i)}\right)=0$$ : different class

<br>

Regularized CE

- $$\begin{gathered}
  \mathcal{L}\left(x_{1}^{(i)}, x_{2}^{(i)}\right)=\mathbf{y}\left(x_{1}^{(i)}, x_{2}^{(i)}\right) \log \mathbf{p}\left(x_{1}^{(i)}, x_{2}^{(i)}\right)+ 
  \left(1-\mathbf{y}\left(x_{1}^{(i)}, x_{2}^{(i)}\right)\right) \log \left(1-\mathbf{p}\left(x_{1}^{(i)}, x_{2}^{(i)}\right)\right)+\boldsymbol{\lambda}^{T} \mid \mathbf{w} \mid ^{2}
  \end{gathered}$$.