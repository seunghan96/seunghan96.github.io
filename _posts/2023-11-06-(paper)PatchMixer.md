---
title: PatchMixer; A Patch-Mixing Architecture for Long-term Time Series Forecasting
categories: [TS, CL]
tags: []
excerpt: ICLR 2024(?)
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# PatchMixer: A Patch-Mixing Architecture for Long-term Time Series Forecasting (ICLR, 2024 (?))

<br>

![figure2](/assets/img/ts/img490.png)

# Contents

0. Abstract

0. Introduction

   

<br>

# Abstract

Transformer vs. CNN

- Transformer: **permutation-invariant** $\rightarrow$ PatchTST

- CNN: **permutation-variant** $\rightarrow$ PatchMixer

<br>

PatchMixer

- only uses **depthwise separable CNN**

- allows to extract

  - local featurers
  - global correlations

  using a single-scale architecture

- dual forecasting heads
  - encompass both "linear & nonlinear" components

<br>

# 1. Introduction

Effectiveness of Transformers in LTSF...??

<br>

PatchTST = ***(1) Patching*** + (2) TST (=Transformer)

- recent works also adopt this "patching" (Zhang & Yan, 2023; Lin et al., 2023)

<br>

Contribution

- (1) propose PatchMixer, based on CNN
- (2) Efficient
  - 3 times faster for inference
  - 2 times faster during training
- (3) SOTA

<br>

# 2. Related Work

## (1) CNN

TCN: dilated causal CNN

SCINet: extract multi-resolutions via binary tree structure

MICN: multi-scale hybrid decomposition & isometric convolution from both local & global perspective

TimesNet: segment sequences into patches

S4: use structured state space model

<br>

## (2) Depthwise Separable Convolution

<br>

## (3) Channel Independence

