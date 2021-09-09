---
title: \[Paper Review\] 08.(improved gan training)Improved Consistency Regularization for GANs
categories: [GAN]
tags: [GAN]
excerpt: 2020, Consistency Regularization, bCR, zCR, ICR
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# \[Paper Review\] 08.Improved Consistency Regularization for GANs

<br>

### Contents

0. Abstract
1. Introduction
2. ICR (Improved Consistency Regularization)

<br>

# 0. Abstract

increase GAN performance by..

$$\rightarrow$$ **forcing a consistency cost on the discriminator!**

<br>

This paper ...

- 1) shows that **CR (Consistency Regularization)** can introduce artifacts into GAN samples
- 2) propose several **modifications to CR procedure**

<br>

# 1. Introduction

### CR-GAN

- **(1) real images & (2) corresponding augmented coutnerparts** are fed into the DISCRIMINATOR

- discriminator is encouraged to produce **similar outpus** for both!

  ( via **auxiliary loss term** )

<br>

### Limitations of CR in CR-GAN

- augmentations are **only applied to real images**, **NOT to generated samples**

  ( imbalanced )

- regularize **ONLY the discriminator**

$$\rightarrow$$ by constraining the mapping from the prior to the generated samples..

can achieve further performance gains!

<br>

### ICR (Improved Consistency Regularization)

applies forms of consistency regularization to the ..

- 1) generated images ( bCR = balanced CR)
- 2) latent vector space ( zCR = latent CR )
- 3) generator

ICR = bCR + zCR

achieve SOTA ( best known FID scores on various GANs )

<br>

# 2. ICR (Improved Consistency Regularization)

Intuition of CR

- encode some prior knowledge to model

  ( = model should produce consistence predictions )

<br>

Augmentations ( or transformations )

- ex) image flipping / rotating / sentence back-translating / adversarial attacks...

<br>

Penalizing inconsistencies via **L2 loss**, **KL-div**

<br>

![figure2](/assets/img/gan/img13.png)