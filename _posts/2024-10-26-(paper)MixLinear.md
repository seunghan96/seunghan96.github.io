---
title: MixLinear; Extereme Low Resource Multivaraite Time Series Forecasting with 0.1K Parameters
categories: [TS]
tags: []
excerpt: Arxiv 2024
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# MixLinear: Extereme Low Resource Multivaraite Time Series Forecasting with 0.1K Parameters

<br>

![figure2](/assets/img/ts2/img201.png)

# Contents

0. Abstract
1. Introduction
2. FAN

3. Experiments



<br>

# 0. Abstract

MixLinear

- Ultra-lightweight MTS forecasting model
  - Designed for resource-constrained devices.
- Effectively captures both 
  - (1) temporal domain
  - (2) frequency domain 

- How? By...
  - (1) Modeling intra-segment and inter-segment variations 
  - (2) Extracting frequency variations 
    - from a low-dimensional latent space in the frequency domain. 

<br>

# 1. Introduction

### MixLinear

- Highly lightweight MTS model

- Efficiently captures the temporal and frequency features from both time and frequency domains. 

- Time domain)
  - Captures intra-segment and inter-segment variations
  - By decoupling channel and periodic information from the trend components
    - Breaking the trend information into smaller segments. 
  
- Frequency domain)
  - By mapping the trend into a latent frequency space 
  
    & reconstructing the trend spectrum.

- Reduction in parameter: from $O\left(n^2\right)$ to $O(n)$ 
  - for $L$-length inputs/outputs 
  - with a known period $w$ 
  - subsequence length $n=\left\lceil\frac{L}{w}\right\rceil$. 

<br>

# 2. MixLinear

## (1) Overview

Key innovation of MixLinear 

= Ability to extract features from **both TIME & FREQ domains**

( while **minimizing the # of parameters** )

<br>

However... combining time and frequency domain models

$\rightarrow$ Significantly increase the parameter scale!

<br>

### MixLinear

**(1) Time Domain Transformation**

- Existing linear models: Apply pointwise transformations

- MixLinear: Captures inter-segment and intra-segment dependencies by splitting the trend into segments

  $\rightarrow$ Significantly reduces the # paarams & enhances the locality

<br>

**(2) Frequency Domain Transformation**

- Focuses on transforming more compact trend components in a lower-dim

  $\rightarrow$ Reduces the model complexity 

<br>

## (2) Time Domain Transformation

Divides the **trend** components ***into smaller segments***

Applies **two linear transformations** to capture 

- (1) **intra**-segment dependencies
- (2) **inter**-segment dependencies.

$\rightarrow$ Significantly reduces the model complexity while enhancing the locality

<br>

Two main subprocesses: 

- a) Trend Segmentation
- b) Segment Transformation

<br>

### a) Trend Segmentation

TS: $X \in \mathbb{R}^L$  ( with the period $w$ )

Extract trend

- step 1) Aggregation
  - Apply a 1D conv( kernel size of $w$ )
  - Aggregate all the information within each period

- step 2) Downsampling
  - Downsample the aggregated series by the period $w$
  - Result: trend = $X_{\text {Trend }} \in \mathbb{R}^n$, where $n=\left\lceil\frac{L}{w}\right\rceil$. 

$\rightarrow$ Effectively decouples the periodic and trend components

( + Zero padding to $X_{\text {Trend }}$ to make $\sqrt{n}$ to be an integer )



Split the trend components $X_{\text {Trend }} \in \mathbb{R}^n$ 

$\rightarrow$ Into smaller trend segments $X_{\text {Seg }} \in \mathbb{R}^{\sqrt{n}}$. 



