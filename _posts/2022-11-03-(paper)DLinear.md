---
title: (paper) Are Transformers Effective for Time Series Forecasting?
categories: [TS]
tags: []
excerpt: 2022
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Are Transformers Effective for Time Series Forecasting? 

<br>

Simplest DMS model, via **LTSF-Linear ( = a temporal linear layer )**

- DMS = Direct Multi-step
- LTSF = Long-Term Time Series Forecasting

<br>

Basic formulation of **LTSF-Linear **

- $\hat{X}_i=W X_i$, 

  - where $W \in \mathbb{R}^{T \times L}$ is a linear layer along the temporal axis

  - ***shares weights across different variates*** 

    & ***does not model any spatial correlations***

- further introduce 2 variants

  - **DLinear**
  - **NLinear**

<br>

# DLinear : Decomposition + Linear

- Decomposition scheme :
  - used in Autoformer and FEDformer
- step 1) decomposes a raw TS  as TREND + REMAINDER
  - TREND = MA kernel
- step 2) 2 one-layer linear layers
  - 1 for TREND
  - 1 for REMAINDER
  - sum up the two features to get the final prediction

<br>

# NLinear : Normalization + Linear

- to boost the performance, when there is a distribution shift in the dataset

- step 1) subtracts the input by the last value of the sequence

  ( = just a simple normalization )

- step 2) input goes through a linear layer

- step 3) subtracted part is added back before making the final prediction

<br>

https://github.com/cure-lab/LTSF-Linear