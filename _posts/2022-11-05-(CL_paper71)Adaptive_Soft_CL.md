---
title: (paper 71) Adaptive Soft Contrastive Learning
categories: [CL, CV, TS]
tags: []
excerpt: 2022
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Adaptive Soft Contrastive Learning

<br>

## Contents

0. Abstract
0. Introduction

2. 

<br>

# 0. Abstract

Contrastive learning

- generally based on instance discrimination tasks

<br>

Problem : presuming all the samples are different

$\rightarrow$ contradicts the natural grouping of similar samples in common visual datasets

<br>

Propose **ASCL (Adaptive Soft Contrastive Learning)**

- adaptive method that introduces **soft inter-sample relations**

- original instance discrimination task $\rightarrow$ multi-instance soft discrimination task
- adaptively introduces **inter-sample relations**

<br>

# 1. Introduction

focus on an inherent deficiency of contrastive learning, **“class collision”**

( = problem of false negatives )

$\rightarrow$ Need to introduce **meaningful inter-sample relations** in contrastive learning.

<br>

ex 1) Debiased contrastive learning

- proposes a theoretical unbiased approximation of contrastive loss with the simplified hypothesis of the dataset distribution
- however, does not address the issue of real false negatives

<br>

ex 2) remove false negatives using progressive mechanism

- NNCLR : define extra positives for each specific view 
  - by ranking and extracting the top-K neighbors in the learned feature space. 
- Co2 : introduces a consistency regularization 
  - enforcing relative distribution consistency of different positive views to all negatives

<br>

ex 3) Clustering-based approaches

- also provide additional positives!

- problems

  - (1) assuming the entire cluster is positive early in the training is problematic
  - (2) clustering has an additional computational cost

  - (3) all these methods rely on a manually set threshold or a predefined number of neighbors

<br>

### ASCL

- **efficient and effective** module for current contrastive learning frameworks

- introduce **inter-sample relations** in an **adaptive** style

- Similarity Distribution

  - use **weakly augmented views** to compute the relative similarity distribution

    & obtain the sharpened soft label information.

  - based on the uncertainty of the similarity distribution, adaptively adjust the weights of the soft labels

- Process

  - (early stage) weights of the soft labels are low and the training of the model will be similar to the original contrastive learning
  - (mature stage) soft labels become more concentrated
    - the model will learn stronger inter-sample relations

<br>

### Main Contributions

- propose a novel adaptive soft contrastive learning (ASCL) method 
  - smoothly alleviates the false negative & over-confidence in the instance discrimination
  - reduces the gap between instance-based learning with cluster-based learning
- show that weak augmentation strategies help to stabilize the CL
- show that ASCL keeps a high learning speed in the initial epochs

<br>

# 2. Related Works

### Introducing Inter-sample relations

how to introduce inter-sample relations into the original instance discrimination task. 

**ex) NNCLR :**

- builds on SimCLR by introducing a memory bank 

- searches for nearest neighbors to replace the original positive samples

**ex) MeanShift :**

- relies on the same idea but builds on BYOL. 

**ex) Co2** 

- an extra regularization term
- to ensure the relative consistency of both positive views with negative samples

**ex) ReSSL** 

- validates that the consistency regularization term itself is enough to learn meaningful representations

<br>

# 3. Adaptive Soft Contrastive Learning

![figure2](/assets/img/cl/img202.png)
