---
title: (paper 43) BEIT
categories: [CL, CV]
tags: []
excerpt: 2022
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# BEIT: BERT Pre-Training of Image Transformers

<br>

## Contents

0. Abstract
1. Methods
   1. Image Representation
   2. Backbone Network : Image Transformer
   3. Pre-Training BEIT : MIM


<br>

# 0. Abstract

self-supervised vision representation model, **BEIT**

( = **Bidirectional Encoder representation from Image Transformers** )

<br>

**BEIT**

- **masked image modeling** task to pretrain vision Transformers
- each image has two views
  - (1) **image patches** (such as 16×16 pixels)
  - (2) **visual tokens** (i.e., discrete tokens)

<br>

Process

- step 1)  **tokenize** the original image into **visual tokens**
- step 2) randomly **mask** some image patches
- step 3) feed them to backbone **Transformer**

<br>

Goal : 

- **recover the original visual tokens**,

  based on the corrupted image patches

<br>

# 1. Methods

![figure2](/assets/img/cl/img88.png)

<br>

BEIT

- encodes input image $$x$$ to **contextualized vector**

- pretrained by the masked image modeling (MIM) task

  ( MIM = recover the masked image patches )

- downstream tasks 
  - ex) image classification, and semantic segmentation
  - append task layers upon pretrained BEIT & fine-tune

<br>

## (1) Image Representation

2 views of representations

- (1) image patch ( serve as **INPUT** )
- (2) visual tokens ( serve as **OUTPUT** )

<br>

### a) Image Patch

image : split into a sequence of patches

- (from) image $$\boldsymbol{x} \in \mathbb{R}^{H \times W \times C}$$
- (to) $$N=H W / P^2$$ patches $$\boldsymbol{x}^p \in \mathbb{R}^{N \times\left(P^2 C\right)}$$ 

<br> 

Image patches $$\left\{\boldsymbol{x}_i^p\right\}_{i=1}^N$$ 

- step 1) flattened into vectors

- step 2) linearly projected

  ( $$\approx$$ word embeddings in BERT )

<br>

### b) Visual Token

represent the image as a sequence of discrete tokens

( = obtained by an "image tokenizer" )

<br>

Tokenize …

- (from) image $$\boldsymbol{x} \in \mathbb{R}^{H \times W \times C}$$
- (to) $$\boldsymbol{z}=\left[z_1, \ldots, z_N\right] \in \mathcal{V}^{h \times w}$$
  - where the vocabulary $$\mathcal{V}=\{1, \ldots, \mid \mathcal{V} \mid \}$$ contains discrete token indices

<br>

Image Tokenizer

- learned by discrete variational autoencoder (dVAE)

- two modules ( during visual token learning )

  - (1) tokenizer : $$q_\phi(\boldsymbol{z} \mid \boldsymbol{x})$$

    - maps image pixels $$\boldsymbol{x}$$ into discrete tokens $$\boldsymbol{z}$$

      ( according to codebook ( =vocab ) )

    - uniform prior

  - (2) decoder : $$p_\psi(\boldsymbol{x} \mid \boldsymbol{z})$$

    - reconstructs the mage $$\boldsymbol{x}$$ based on the visual tokens $$\boldsymbol{z}$$

    - Reconstruction objective : $$\mathbb{E}_{\boldsymbol{z} \sim q_\phi(\boldsymbol{z} \mid \boldsymbol{x})}\left[\log p_\psi(\boldsymbol{x} \mid \boldsymbol{z})\right]$$

      ( discrete? use **Gumbel Softmax Trick** )

<br>

Details :

- \# of visual tokens = \# of image patches
- vocab size : $$ \mid \mathcal{V} \mid =8192$$

<br>

## (2) Backbone Network : Image Transformer

( use the standard Transformer as the backbone )

<br>

a) Input ( of Transformer ) : 

- sequence of image patches $$\left\{\boldsymbol{x}_i^p\right\}_{i=1}^N$$

  ( $$N$$ = number of patches )

<br>

b) Embeddings :

- $$\left\{\boldsymbol{x}_i^p\right\}_{i=1}^N$$ are linearly projected to $$\boldsymbol{E} \boldsymbol{x}_i^p$$
  - where $$\boldsymbol{E} \in \mathbb{R}^{\left(P^2 C\right) \times D}$$
- add learnable 1d positional embeddings : $$\boldsymbol{E}_{\text {pos }} \in \mathbb{R}^{N \times D}$$
- final output embedding : $$\boldsymbol{H}_0=\left[\boldsymbol{e}_{[\mathrm{S}]}, \boldsymbol{E} \boldsymbol{x}_i^p, \ldots, \boldsymbol{E} \boldsymbol{x}_N^p\right]+\boldsymbol{E}_{\text {pos }}$$

<br>

c) Encoder :

- contains $$L$$ layers of Transformer blocks

  - $$\boldsymbol{H}^l=\operatorname{Transformer}\left(\boldsymbol{H}^{l-1}\right)$$.

- **output vectors of the last layer : $$\boldsymbol{H}^L=\left[\boldsymbol{h}_{[\mathrm{s}]}^L, \boldsymbol{h}_1^L, \ldots, \boldsymbol{h}_N^L\right]$$**

  ( $$\boldsymbol{h}_i^L$$ : vector of the $$i$$-th patch )

  $$\rightarrow$$ encoded representations for the image patches

<br>

## (3) Pre-Training BEIT : MIM

randomly mask some % of image patches

& predict the visual tokens ( corresponding to the masked patches )

<br>

Notation

- Input image : $$\boldsymbol{x}$$
  - $$N$$ image patches : $$\left(\left\{\boldsymbol{x}_i^p\right\}_{i=1}^N\right)$$
  - $$N$$ visual tokens : $$\left(\left\{z_i\right\}_{i=1}^N\right)$$
- Masked positions : $$\mathcal{M} \in\{1, \ldots, N\}^{0.4 N}$$
  - randomly mask approximately $$40 \%$$ image patches

<br>

Replace the masked patches with a learnable embedding $$e_{[M]} \in \mathbb{R}^D$$. 

$$\rightarrow$$ corrupted image patches : $$x^{\mathcal{M}}=\left\{\boldsymbol{x}_i^p: i \notin \mathcal{M}\right\}_{i=1}^N \bigcup\left\{\boldsymbol{e}_{[M]}: i \in \mathcal{M}\right\}_{i=1}^N$$

<br>

![figure2](/assets/img/cl/img89.png)

<br>

$$x^{\mathcal{M}}$$ are then fed into the $$L$$-layer Transformer

$$\rightarrow$$ final hidden vectors : $$\left\{\boldsymbol{h}_i^L\right\}_{i=1}^N$$

( = regarded as encoded representations of the input patches )

<br>

Classification ( with softmax classifier )

- classify for each masked position $$\left\{\boldsymbol{h}_i^L: i \in \mathcal{M}\right\}_{i=1}^N$$
- $$p_{\mathrm{MIM}}\left(z^{\prime} \mid x^{\mathcal{M}}\right)=\operatorname{softmax}_{z^{\prime}}\left(\boldsymbol{W}_c \boldsymbol{h}_i^L+\boldsymbol{b}_c\right)$$.
  - $$x^{\mathcal{M}}$$ : corrupted image
  - $$\boldsymbol{W}_c \in \mathbb{R}^{ \mid \mathcal{V} \mid  \times D}$$ and $$\boldsymbol{b}_c \in \mathbb{R}^{ \mid \mathcal{V} \mid }$$

<br>

Pre-training objective 

- maximize the log-likelihood of the correct visual tokens $$z_i$$ given the corrupted image:

- $$\max \sum_{x \in \mathcal{D}} \mathbb{E}_{\mathcal{M}}\left[\sum_{i \in \mathcal{M}} \log p_{\operatorname{MIM}}\left(z_i \mid x^{\mathcal{M}}\right)\right]$$.

<br>


