---
title: (paper) A Time Series is Worth 64 Words ; Long-term Forecasting with Transformers
categories: [TS,CL]
tags: []
excerpt: 2023
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# A Time Series is Worth 64 Words : Long-term Forecasting with Transformers

( https://openreview.net/pdf?id=Jbdc0vTOcol )

<br>

## Contents

0. Abstract
1. Introduction
2. Related Works
3. 

<br>

# 0. Abstract

Propose an efficient design of Transformer-based models for MTS forecasting & SSL

<br>

2 key components :

- (1) segmentation of TS into subseries-level patches
  - served as input tokens to Transformer
- (2) channel-independence 
  - each channel contains a single univariate TS
  - shares the same embedding and Transformer weights across all the series

<br>

3 benefits of patching design 

- (1) **local semantic information** is retained in the embedding
- (2) **computation and memory usage** of the attention maps are quadratically reduced
- (3) can attend **longer history**

<br>

channel-independent patch time series Transformer (PatchTST) 

- improve the (1) long-term forecasting accuracy 
- apply our model to (2) SSL tasks

<br>

# 1. Introduction

Recent paper (Zeng et al., 2022)  : ***very simple linear model can outperform all of the previous models*** on a variety of common benchmarks

<br>

This paper propose **a channel-independence patch time series Transformer (PatchTST)** model that contains 2 key designs :

### (1) Patching

- TS forecasting :  need to understand the correlation between data in each different time steps

  - single time step does not have semantic meaning 

    $$\rightarrow$$ extracting local semantic information is essential

  - However …. most of the previous works only use **point-wise** input tokens

- This paper : enhance the locality & capture comprehensive semantic information that is not available in point-level

  $$\rightarrow$$ **by aggregating time steps into subseries-level patches**

<br>

### (2) Channel-independence

- MTS is multi-channel signal

  - each Transformer input token can be represented by data from either a single or multiple channels

- different variants of the Transformer depending on the design of input tokens

- Channel-mixing :

  - input token takes the vector of all time series features

    & projects it to the embedding space to mix information

- Channel-independence :
  - each input token only contains information from a single channel

<br>

# 2. Proposed Method

## (1) Model Structure

MTS with lookback window $$L$$ : $$\left(\boldsymbol{x}_1, \ldots, \boldsymbol{x}_L\right)$$ 

- each $$\boldsymbol{x}_t$$ : vector of dimension $$M$$

Goal : forecast $$T$$ future values $$\left(\boldsymbol{x}_{L+1}, \ldots, \boldsymbol{x}_{L+T}\right)$$

<br>

### a) Architecture

encoder : vanilla Transformer

![figure2](/assets/img/cl/img323.png)

<br>

### b) Forward Process

$$\boldsymbol{x}_{1: L}^{(i)}=\left(x_1^{(i)}, \ldots, x_L^{(i)}\right)$$ : $$i$$-univariate TS of length $$L$$

- where $$i= 1, \cdots M$$

<br>

Input : $$\left(\boldsymbol{x}_1, \ldots, \boldsymbol{x}_L\right)$$ 

- split to $$M$$ univariate TS  $$\boldsymbol{x}^{(i)} \in \mathbb{R}^{1 \times L}$$

- each of them is fed independently into the Transformer backbone

  ( under channel-independence setting )

<br>

Output :  $$\hat{\boldsymbol{x}}^{(i)}=\left(\hat{x}_{L+1}^{(i)}, \ldots, \hat{x}_{L+T}^{(i)}\right) \in \mathbb{R}^{1 \times T}$$ 

- forecasting horizon : $$T$$

<br>

### c) Patching

Input : univariate time series $$\boldsymbol{x}^{(i)}$$ 

- divided into patches ( either overlapped or non-overlapped )
  - patch length = $$P$$
  - stride = $$S$$

<br>

Output : sequence of patches $$\boldsymbol{x}_p^{(i)} \in \mathbb{R}^{P \times N}$$

- $$N=\left\lfloor\frac{(L-P)}{S}\right\rfloor+2$$ : number of patches  
- pad $$S$$ repeated numbers of the last value $$x_L^{(i)} \in \mathbb{R}$$ to the end

<br>

Result : number of input tokens can reduce from $$L$$ to approximately $$L / S$$. 

- memory usage & computational complexity of the attention map : quadratically decreased by a factor of $$S$$

![figure2](/assets/img/cl/img324.png)

<br>

### d) Loss Function : MSE

Loss in each channel :  $$\left\|\hat{\boldsymbol{x}}_{L+1: L+T}^{(i)}-\boldsymbol{x}_{L+1: L+T}^{(i)}\right\|_2^2$$ 

Total Loss : $$\mathcal{L}=\mathbb{E}_{\boldsymbol{x}} \frac{1}{M} \sum_{i=1}^M\left\|\hat{\boldsymbol{x}}_{L+1: L+T}^{(i)}-\boldsymbol{x}_{L+1: L+T}^{(i)}\right\|_2^2 $$

<br>

### e) Instance Normalization

help mitigating the **distribution shift effect** ( between the training and testing data )

simply normalizes each time series instance $$\boldsymbol{x}^{(i)}$$ with N(0,1)

$$\rightarrow$$ normalize each $$\boldsymbol{x}^{(i)}$$ before patching & scale back at prediction

<br>

## (2) Representation Learning

Propose to apply PatchTST to obtain useful representation of the multivariate time series

- via masking & reconstructing

<br>

Apply the MTS to transformer ( each input token is a vector $$\boldsymbol{x}_i$$ )

Masking : placed randomly within each TS and across different series

<br>

2 potential issues 

- (1) masking is applied at the level of single time steps
  - masked values : can be easily inferred by interpolating
- (2) design of the output layer for forecasting task can be troublesome
  - parameter matrix $$W$$ : $$(L \cdot D) \times(M \cdot T)$$
    - $$L$$ : time length
    - $$D$$ : dimension of $$\boldsymbol{z}_t \in \mathbb{R}^D$$ corresponding to all $$L$$ time steps
    - $$M$$ : TS with $$M$$ variable
    - $$T$$ : prediction horizon

<br>

PatchTST overcome these issues

- Instead of prediction head …. attach $$D \times P$$ linear layer
- Instead of overlapping patches …. use non-overlapping patches
  - ensure observed patches do not contain information of the masked ones
  - select a subset of the patch indices uniformly at random 

<br>

etc )

- trained with MSE loss to **reconstruct the masked patches**

- each TS will have its own latent representation 
  - cross-learned via **shared weight**
  - allow the pre-training data to contain **different \# of TS**

<br>

# 4. Experiments

## (1) Long Term TS Forecasting

### a) Datasets

![figure2](/assets/img/cl/img325.png)

<br>

### b) Results

![figure2](/assets/img/cl/img326.png)

