---
title: (paper 3) MoCo v2
categories: [CL]
tags: []
excerpt: 2020
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Improved Baselines with Momentum Contrastive Learnins

<br>

### Moco v2 = MoCo v1 + SimCLR 

<br>

### Moco v1

- **UNsupervised** visual representation learning

- **dictionary look-up** perspective

  $$\rightarrow$$ build a **dynamic dictionary** ( with a queue = FIFO )

- **moving-averaged** encoder

<br>

### SimCLR

- (1) large batch size for lots of negative samples
- ***(2) stronger augmentation***
- ***(3) MLP projection head***

<br>

MoCo v2 = Moco V1 + SimCLR (2) & (3)

<br>

![figure2](/assets/img/cl/img8.png)

<br>

