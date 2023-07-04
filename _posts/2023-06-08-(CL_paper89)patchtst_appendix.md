---
title: (paper 89) PatchTST Experiments
categories: [CV, CL, SEMI]
tags: []
excerpt: 2023
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# PatchTST Experiments

<br>

## Contents

0. Abstract
1. Introduction
   


<br>

# 1. Supervised Learning

## (1) Datasets

8 datasets for LTSF

- large dataset : weather, traffic, electricity
- small dataset : ILI + 4 ETT datasets

![figure2](/assets/img/ts/img416.png)

<br>

## (2) Settings

Prediction Length :

- ILI : [24,36,48,60]
- others : [96,192,336,720]

<br>

Lookback Window : 

- DLinear : 336
- xx-former : 96

<br>

## (3) Model Variants 

**2 versions of PatchTST**

- PatchTST/64 
  - number of patches = 64
  - lookback window = 512
  - patch length = 16 
  - stride = 8
- PatchTST/42
  - number of patches = 42
  - lookback window = 336
  - patch length = 16 
  - stride = 8

<br>

## (4) Results

( Compared with DLinear )

- outperform especially in **LARGE datasets** ( Weather, Traffic, Electricity) & **ILI**

![figure2](/assets/img/ts/img417.png)

<br>

# 2. Self-Supervised Learning

Details

- **NON-overlapped patch**
- masking ratio : 40% ( with zero )

<br>

Input settings

- input size = 512
- number of patches = 42
- patch length & stride = 12

<br>

Procedure

- step 1) Pretraining 100 epochs
- step 2)
  - 2-1) Linear Probing ( 20 epochs )
  - 2-2) E2E fine-tuning ( linear probing 10 epochs + E2E 20 epochs )

<br>

Results

![figure2](/assets/img/ts/img418.png)

- Fine-tuning > Linear Probing = Supervised

<br>

# 3. SL vs SSL

Large

Fine-tuning > Linear Probing = Supervised

