---
title: (paper 21) BigBiGAN
categories: [CL, CV]
tags: []
excerpt: 2019
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Large Scale Adversarial Representation Learning

<br>

## Contents

0. Abstract
0. BigBiGAN
   0. Encoder $$\mathcal{E}$$
   0. Joint Discriminator $$\mathcal{D}$$ 


<br>

# 0. Abstract

BigBiGAN

- builds upon BiGAN model
- extend it to representation learning, by…
  - (1) adding an encoder
  - (2) modifying the discriminator

<br>

# 1. BigBiGAN

![figure2](/assets/img/cl/img49.png)

<br>

## (1) Encoder $$\mathcal{E}$$

models the inverse conditional distn $$P(\mathbf{z} \mid \mathbf{x})$$

( = predicting **latents $$\mathbf{z}$$ given $$\mathbf{x}$$** )

<br>

## (2) Joint Discriminator $$\mathcal{D}$$

- takes as input **data-latent pairs $$(\mathbf{x}, \mathbf{z})$$**

- learns to discriminate between pairs from (1) vs (2)
  - (1) data distribution and encoder $$\left(\mathbf{x} \sim P_{\mathbf{x}}, \hat{\mathbf{z}} \sim \mathcal{E}(\mathbf{x})\right)$$
  - (2) generator and latent distribution $$\left(\hat{\mathbf{x}} \sim \mathcal{G}(\mathbf{z}), \mathbf{z} \sim P_{\mathbf{z}}\right)$$

<br>

Loss function : 

$$\min _{\mathcal{G} \mathcal{E}} \max _{\mathcal{D}}\left\{\mathbb{E}_{\mathbf{x} \sim P_{\mathbf{x}}, \mathbf{z} \sim \mathcal{E}_{\Phi}(\mathbf{x})}[\log (\sigma(\mathcal{D}(\mathbf{x}, \mathbf{z})))]+\mathbb{E}_{\mathbf{z} \sim P_{\mathbf{z}}, \mathbf{x} \sim \mathcal{G}_{\Phi}(\mathbf{z})}[\log (1-\sigma(\mathcal{D}(\mathbf{x}, \mathbf{z})))]\right\}$$.
