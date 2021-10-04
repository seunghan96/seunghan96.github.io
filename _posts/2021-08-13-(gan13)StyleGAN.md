---
title: \[Paper Review\] 13.(G arch) A Style-Based Generator Architecture for Generative Adversarial Networks
categories: [GAN]
tags: [GAN]
excerpt: 2019, StyleGAN
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# \[Paper Review\] 13. A Style-Based Generator Architecture for Generative Adversarial Networks

### Contents

0. Abstract
1. Introduction
2. Style-based Generator
3. Properties of Style-based Generator
   1. Style Mixing
   2. Stochastic Variation
4. Disentanglement studies

<br>

# 0. Abstract

New architecture of GAN

- a) unsupervised separation of high-level attributes
  - ex) pose, identity of face..
- b) stochastic variation in the generated images
  - ex) freckles, hairs ....
- c) enables intuitive, scale-specific control

<br>

# 1. Introduction

Problems of $$G$$

- operate as BLACK-BOX models
- properties of the latent space are poorly understood

<br>

Motivated by **style-transfer**...

$$\rightarrow$$ redesign "$$G$$ architecture" in a way that **exposes novel ways to control image synthesis process**

<br>

Our $$G$$...

- embeds $$z$$ into intermediate latent space ( = $$g(z)$$ )

- $$g(z)$$ is free from restriction

  $$\rightarrow$$ allowed to be disentangled

<br>

Propose 2 new automated metrics

- **1) perceptual path length**
- **2) linear separability**

for quantifying these aspects of $$G$$

<br>

# 2. Style-based Generator

(Traditionally)

- latent code $$\rightarrow$$ input layer

(Proposed)

- omitting the input layer altogether....
- start from a learned constant instead!

- use "intermediate latent space"
- then, controls the $$G$$ through **AdaIN** at each convolution layer
  - $$A$$ : learned affine transform
    - specialize $$w$$ to styles $$\mathbf{w}$$ to styles $$\mathbf{y}=\left(\mathbf{y}_{s}, \mathbf{y}_{b}\right)$$ that control AdaIN
  - $$B$$ : learned per-channel scaling factors to the noise input

![figure2](/assets/img/gan/img35.png)

<br>

### AdaIN

$$\operatorname{AdaIN}\left(\mathbf{x}_{i}, \mathbf{y}\right)=\mathbf{y}_{s, i} \frac{\mathbf{x}_{i}-\mu\left(\mathbf{x}_{i}\right)}{\sigma\left(\mathbf{x}_{i}\right)}+\mathbf{y}_{b, i}$$.

- each feature map $$\mathbf{x}_{i}$$ is normalized separately

<br>

Compared to style transfer...

- compute the spatially invariant style $$\mathbf{y}$$ from vector $$\mathbf{w}$$ , instead of example image
- provide $$G$$ with direct means to generate **stochastic detail** by introducing **explicit noise**

<br>

# 3. Properties of Style-based Generator

proposed $$G$$ : able to **control** the image synthesis,

- via **scale-specific** modifications to the **styles**

<br>

1) Mapping Network & Affine transformation

​	= draw samples for each style from a learend distn

2) Synthesis network

 	= generate novel images, based on a **collection of styles**

<br>

## 3-1. Style Mixing

- to encourage the styles to localize...  use **mixing regularization**

- use **2** random latent code!

  - when generating image, switch from one to another at certain time

- ex) $$\mathbf{z}_1$$ & $$\mathbf{z}_2$$ through mapping network,

  make $$\mathbf{w}_1$$ & $$\mathbf{w}_2$$ control the styles

- this regularization **prevents** the network from assuming that **adjacent styles are correlated**

  - ex) make **hair style** & **hair color** decorrelated

<br>

images synthesized by **mixing two latent codes at various scales**

![figure2](/assets/img/gan/img38.png)

![figure2](/assets/img/gan/img39.png)

<br>

## 3-2. Stochastic Variation

ex) exact placement of hairs, stubble, freckles, skin pores...

how to make variation?

$$\rightarrow$$ adding **per-pixel noise AFTER each convolution**

![figure2](/assets/img/gan/img37.png)

<br>

# 4. Disentanglement studies

by NOT DIRECTLY making image from $$z$$....

( = **by using MAPPING NETWORK...** )

$$\rightarrow$$ $$\mathbf{w}$$ need not follow fixed distn! 

**use more flexible intermediate latent space, thus easier to control visual attribute!**

![figure2](/assets/img/gan/img36.png)