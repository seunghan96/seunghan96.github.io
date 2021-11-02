---
title: \[Paper Review\] 24.(Gan Inversion) Exploiting Spatial Dimensions of Latent in GAN for Real-time Image Editing
categories: [GAN]
tags: [GAN]
excerpt: 2021, StyleMapGAN
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# \[Paper Review\] 24. Exploiting Spatial Dimensions of Latent in GAN for Real-time Image Editing

<br>

### Contents

0. Abstract
1. Introduction
2. Related Work
   1. Optimization-based editing methods
   2. Learning-based editing methods
   3. Local editing methods
3. StyleMapGAN
   1. Stylemap-based generator
   2. Training procedure and losses
   3. Local Editing

<br>

# 0. Abstract

![figure2](/assets/img/gan/img62.png)

<br>

Editing **REAL** images with GAN : suffer from...

- [ problem 1 ] 

  **time-consuming optimization** for projecting real $$\rightarrow$$ latent

- [ problem 2 ]

  **inaccurate embedding** through an encoder

<br>

Propose **StyleMapGAN**

- intermediate latent space has **SPATIAL** dimensions
- **spatially variant modulation** replaces AdaIN

<br>

# 1. Introduction

still challenging to apply manipulations to **REAL IMAGES**

- since GAN ***lacks an inverse mapping*** from image back to latent code

<br>

[ Manipulating REAL images ]

**(1) image-to-image translation**

- synthesize an output image, given a user's input directly
- problem : need pre-defined tasks & heavy supervision



**(2) pretrained GAN**

- directly optimize the latent code for eadch image



**(3) train extra encoder**

- more practical

- project an image into its corresponding latent code

- single feed-forward

- BUT, **low fidelity of projected images**

  due to **"absence of spatial dimension" in the latent space**

<br>

### StyleMapGAN

- exploits style map, a novel representation of latent space

- vector-based representation (X)

  tensor with explicit spatial dimensions (O)

<br>

# 2. Related Work

## 1) Optimization-based editing methods

iteratively update latent vector of pre-trained GANs

examples)

- Image2StyleGAN
- In-DomainGAN
- Neural Collage, pix2latent

<br>

but this paper **exploits an encoder**,

**which is faster** than the above methods!

<br>

## 2) Learning-based editing methods

train an **extra encoder** to DIRECTLY infer the latent code

examples)

- ALI
- BiGAN
- ALAE

<br>

FAST, but all those methods lack spatial dimensions!

<br>

## 3) Local editing methods

editing specific parts

examples)

- Editing in Style
- Structured Noise
- SEAN

<br>

# 3. StyleMapGAN

GOAL :

- **project images to latent space**
- with an **encoder**
- in **real-time**
- and **locally manipulate** images on **latent space**

<br>

Propose StyleMapGAN...

- 1) intermediate latent space with spatial dimensions
- 2) spatially variant modulation based on the stylemap

<br>

## 1) Stylemap-based generator

![figure2](/assets/img/gan/img64.png)

<br>

Spatial dimensions

- much more effective at inference
- enables local editing

<br>

Affine Transform

- produces parameters for modulation, regarding the resized stylemaps

- modulation operation of the i-th layer :

  $$h_{i+1}=\left(\gamma_{i} \otimes \frac{h_{i}-\mu_{i}}{\sigma_{i}}\right) \oplus \beta_{i}$$.

<br>

Remove per-pixel noise

- per-pixel noise : extra source of spatially varying inputs
- BUT stylemap already provides spatially varying inputs!

<br>



## 2) Training procedure and losses

![figure2](/assets/img/gan/img65.png)

- F : mapping network
- G : synthesis network with stylemap resizer
- E : encoder
- D : discriminator

<br>

## 3) Local Editing

GOAL

- transplant some parts of reference image to an original image, w.r.t a mask

- project **original & reference** image through the encoder

  and obtain stylemaps $$\mathrm{w}$$ and $$\widetilde{\mathrm{w}}$$

- editied style map $$\ddot{\mathbf{w}}$$ :

  $$\ddot{\mathbf{w}}=\mathbf{m} \otimes \widetilde{\mathbf{w}} \oplus(1-\mathbf{m}) \otimes \mathbf{w}$$.

<br>

![figure2](/assets/img/gan/img63.png)