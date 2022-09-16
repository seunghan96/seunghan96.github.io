---
title: (paper 41) ConvMAE ; Masked Convolution Meets Masked Autoencoders
categories: [CL, CV]
tags: []
excerpt: 2022
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# ConvMAE : Masked Convolution Meets Masked Autoencoders

<br>

## Contents

0. Abstract
1. Approach
   1. Masked Autoencoders (MAE)
   2. ConvMAE

<br>

# 0. Abstract

ConvMAE framework

- **multi-scale hybrid convolution-transformer** can learn more discriminative representations via the **mask auto-encoding** scheme

- directly using the original masking strategy : heavy computational cost

  $$\rightarrow$$ solution ) adopt the **masked convolution** 

- simple **block-wise masking strategy** for computational efficiency
- propose to more **directly supervise the multi-scale features of the encoder** to boost multi-scale features

<br>

# 1. Approach

## (1) Masked Autoencoders (MAE)

Details :

- **simple, but strong & scalable** pretraining framework for learning **visual representations**

- **self-supervised method for pretraining ViT**

  ( by **reconstructing masked RGB patches**, from visible patches )

- consists of transformer-based **ENCODER & DECODER**
  - ENCODER ) only **visible patches** are fed
  - DECODER ) learnable **mask tokens** are processed

![figure2](/assets/img/cl/img81.png)

<br>

## (2) ConvMAE

ConvMAE = simple & effictive derivative of MAE

( + modifications on the **encoder design** & **masking strategy** )

<br>

Goal of ConvMAE :

- (1) learn **discriminative multi-scale visual representations**

- (2) **prevent pretraining-finetuning discrepency**

![figure2](/assets/img/cl/img82.png)

<br>

### a) Hybrid Convolution-transformer Encoder

encoder consists of **3 stages**

- with output **spatial resolutions of $$\frac{H}{4} \times \frac{W}{4}, \frac{H}{8} \times \frac{W}{8}, \frac{H}{16} \times \frac{W}{16}$$**

- **[ 1 & 2 stage ]**

  - use **convolution blocks** to transform the **inputs to token embeddings**

    - $$E_1 \in \mathbb{R}^{\frac{H}{4} \times \frac{W}{4} \times C_1}$$ & $$E_2 \in \mathbb{R}^{\frac{H}{8} \times \frac{W}{8} \times C_2}$$

  - follow the design principle of the transformer block 

    ( by only replacing the self-attention operation with the $$5 \times 5$$ depthwise convolution )

- **[ 3 stage ]**

  - use **self-attention blocks** to obtain token embeddings
    - $$E_3 \in \mathbb{R}^{\frac{H}{16} \times \frac{W}{16} \times C_3}$$.

- between every stage... 

  $$\rightarrow$$ **stride-2 convolutions** are used to downsample the tokens

<br>

### b) Block-wise Masking with Masked Convolutions

![figure2](/assets/img/cl/img83.png)