---
title: (paper 24) Selfie
categories: [CL, CV]
tags: []
excerpt: 2019
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Selfie : Self-supervised Pretraining for Image Embedding

<br>

## Contents

0. Abstract
0. Method
   0. Pretraining Details
   0. Attention Pooling



<br>

# 0. Abstract

introduce a pretraining technique called ***Selfie***

( = SELF-supervised Image Embedding )

- generalizes the concept of **masked language modeling** of BERT to image
- learns to select the **correct patch**

<br>

# 1. Method

2 stage

- (1) pre-training
- (2) fine-tuning

<br>

(1) pre-training stage

- $$P$$ : patch processing network

- produce 1 feature vector per patch ( for both ENC & DEC )

- Encoder

  - feature vectors are **pooled** by ***attention pooling network $$A$$***

    $$\rightarrow$$ produce a single vector $$u$$

- Decoder

  - no pooling
  - feature vectors are sent directly to the computation loss

- Encoder & Decoder : jointly trained

<br>

(2) fine-tuning stage

- goal : improve ResNet-50

  $$\rightarrow$$ pretrain the first 3 blocks of this architecture ( = $$P$$ )

## (1) Pretraining Details

- use a part of the input image to predict the rest of the image

![figure2](/assets/img/cl/img52.jpeg)

- ex) Patch 1,2,5,6,7,9 : sent to **Encoder**
- ex) Patch 3,4,8 : sent to **Decoder**

<br>

### a) Patch Sampling method

image size 32x32 $$\rightarrow$$ patch size = 8x8

image size 224x224 $$\rightarrow$$ patch size = 32x32

<br>

### b) Patch processing network

focus on improving ResNet-50

use it as the **path processing network** $$P$$

<br>

### c) Efficient implementation of mask prediction

for efficiency… decoder is implemented to predict **multiple correct patches** for **multiple locations** at the ***same time***

<br>

## (2) Attention Pooling

attention pooling network : $$A$$

<br>

### a) Transformer as pooling operation

notation

- patching processing network : $$P$$

- input vectors : $$\left\{h_1, h_2, \ldots, h_n\right\}$$

  $$\rightarrow$$ pool them to single vector $$u$$

<br>

attention pooling

- $$u, h_1^{\text {output }}, h_2^{\text {output }}, \ldots, h_n^{\text {output }}=\operatorname{TransformerLayers}\left(u_o, h_1, h_2, \ldots, h_n\right)$$.

  ( use only $$u$$ as the pooling fresult! )

<br>

### b) Positional embedding

( image size 32x32 ) : 16 patches ( of size 8x8 )

( image size 224x224 ) : 49 patches ( of size 32x32 )

$$\rightarrow$$ instead of learning 49 positional embeddings … only need to learn 7+7 (=14) embeddings

