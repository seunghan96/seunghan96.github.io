---
title: \[Paper Review\] 23.(Gan Inversion) Image2StyleGAN ; How to Embed Images Into the StyleGAN Latent Space?
categories: [GAN]
tags: [GAN]
excerpt: 2019, Image2StyleGAN 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# \[Paper Review\] 23. Image2StyleGAN : How to Embed Images Into the StyleGAN Latent Space?

<br>

### Contents

0. Abstract
1. Introduction
2. Related Work
   1. Latent Space Embedding
   2. Perceptual Loss & Style Transfer
3. What Images can be embedded into StyleGAN latent space?
   1. Embedding result
   2. How Robust is the Embedding of Face Images?
   3. Which Latent Space to Choose?
4. How Meaningful is the Embedding?
   1. Morphing
   2. Style Transfer
   3. Expression Transfer

<br>

# 0. Abstract

propose an efficient algorithm to..

- embed a given image into latent space of StyleGAN
- enables **semantic image editing operations**

<br>

# 1. Introduction

Background

- quality of GAN increased rapidly!

- StyleGAN makes use of intermediate $$W$$ latent space

Q) ***Is it possible to embed a given photograph into the GAN latent space?***

<br>

This paper builds an embedding algorithm,

that **map a given image $$I$$ in the latent space of StyleGAN**

<br>

Analyze the quality of embedding to see if it is **semantically meaningful**

Propose 3 basic operations on vectors in the latent space

- 1) linear interpolation ........... morphing
- 2) crossover .............. style transfer
- 3) adding a vector & scaled difference vector ......... expression transfer

<br>

# 2. Related Work

## 1) Latent Space Embedding

2 approaches to embed instance..... image space $$\rightarrow$$ latent space

- 1) learn an encoder ( ex. VAE )
  - fast, but problem in generalizaing
- 2) select a random initial latent code & optimize it
  - general & stable solution

<br>

## 2) Perceptual Loss & Style Transfer

low-level similarity

- measured in pixel space with L1/L2 loss

<br>

high-level similarity

- measured between images perceptually
- perceptual loss

<br>

Different layers of VGG net, 

- extract features at different scales, 
- and can be **separated into content & style**

<br>

# 3. What Images can be embedded into StyleGAN latent space?

## 1) Embedding Result

![figure2](/assets/img/gan/img55.png)

- embedded Obama faces is of very high perceptual quality

<br>

## 2) How Robust is the Embedding of Face Images?

### Affine Transformation

![figure2](/assets/img/gan/img56.png)

- very sensitive to affine transformations

  ( translation, resizing, rotation )

- implies that learned representations are still **scale and position dependent**

<br>

### Embedding Defective Images

![figure2](/assets/img/gan/img57.png)

- quite robust to defects in images
- does not inpaint the missing information

<br>

## 3) Which Latent Space to Choose?

there are multiple latent spaces in StyleGAN

2 candidates

- 1) initial latent space $$Z$$
- 2) intermediate latent space $$W$$

<br>

Not easily possible to embed $$W$$ or $$Z$$ directly

$$\rightarrow$$ propose to embed into extended latent space $$W^{+}$$

- $$W^{+}$$ : concatenation of 18 different 512-dim $$w$$ vectors

<br>

Example

- embedding in to $$W$$ directly, does not give good results

![figure2](/assets/img/gan/img58.png)

<br>

# 4. How Meaningful is the Embedding?

propose 3 tests to evaluate, 

if an embedding is semantically meaningful

- conducted by simple **latent code manipulations of vectors $$w_i$$**
  - 1) morphing
  - 2) expression transfer
  - 3) style transfer

<br>

## 1) Morphing

- given 2 embedded images with respective latent vectors $$w_1,w_2$$,

- morphing : linear interpolation

  $$w=\lambda w_{1}+(1-\lambda) w_{2}, \lambda \in(0,1)$$.

<br>

![figure2](/assets/img/gan/img59.png)

- generates high-quality morphing
- but fails on non-face images

<br>

## 2) Style Transfer

- given 2 embedded images with respective latent vectors $$w_1,w_2$$,
- style transfer is computed by **crossover operation**

![figure2](/assets/img/gan/img60.png)

<br>

## 3) Expression Transfer

- given 3 embedded images with respective latent vectors $$w_1,w_2$$,$$w_3$$
- $$w = w_1+ \lambda (w_3 - w_2)$$.
  - $$w_1$$ : latent code of **target image**
  - $$w_2$$ : neutral expression of **source image**
    - ex) expressionless face of Tom
  - $$w_3$$ : more distinct expression
    - ex) smiling face of Tom

![figure2](/assets/img/gan/img61.png)