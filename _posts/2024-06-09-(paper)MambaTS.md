---
title: MambaTS: Improved Selective SSM for LTSF
categories: [TS,MAMBA]
tags: []
excerpt: arxiv 2024
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting

<br>

![figure2](/assets/img/ts2/img135.png)



# Contents

0. Abstract

<br>

# 0. Abstract

Limitations of current Mamba in LTSF

### MambaTS

- Propose 4 targeted improvements

- (1) Variable scan along time 
  - to arrange the historical information of all the variables together. 
- (2) Temporal Mamba Block (TMB)
  - suggest that causal convolution in Mamba is not necessary for LTSF 
  - incorporate a dropout mechanism for selective parameters of TMB to mitigate model overfitting. Moreover, we tackle the issue of variable scan order sensitivity by introducing variable permutation training. We further propose variable-aware scan along time to dynamically discover variable relationships during training and decode the optimal variable scan order by solving the shortest path visiting all nodes problem during inference. Extensive experiments conducted on eight public datasets demonstrate that MambaTS achieves new state-of-the-art performance.
