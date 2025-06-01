---
title: How Can Time Series Analysis Benefit From Multiple Modalities? A Survey and Outlook - Part 2
categories: [MULT, TS]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# How Can Time Series Analysis Benefit From Multiple Modalities? A Survey and Outlook - Part 2

<br>

# Contents

0. Abstract
1. Introduction
2. Background and Taxonomy
   1. Taxonomy
   2. Background
3. TimeAsX: Resuing Foundation Models of Other Modalities for Efficient TSA
   1. Time As Text
   2. **Time As Image**
   3. **Time As Other Modalities**
   4. **Domain-Specific TS Works**

<br>

# 3. TimeAsX: Resuing Foundation Models of Other Modalities for Efficient TSA

![figure2](/assets/img/ts/img754.png)

<br>

## (2) Time As Image

Quite natural! Similar to how humans perceive patterns!

<br>

### a) Line-graphs

- Most popular way to convert TS2image
- To use vision foundational models (e.g., ViT)
  - E.g., VLMs for anomaly detection [200, 237] and classification [37]. 
- Examples) [137, 218, 200, 237, 37]
  - [137] *ViTime: A Visual Intelligence-Based Foundation Model for Time Series Forecasting* https://www.arxiv.org/pdf/2407.07311v3
  - [218] *Time Series as Images: Vision TransformerforIrregularly Sampled Time Series* https://www.arxiv.org/pdf/2303.12799
  - [200] *Can Multimodal LLMs Perform Time Series Anomaly Detection?* https://www.arxiv.org/pdf/2502.17812
  - [237] *See it, Think it, Sorted: Large Multimodal Models are Fewshot Time Series Anomaly Analyzers*. https://www.arxiv.org/pdf/2411.02465
  - [37] *Plots Unlock Time-Series Understanding in Multimodal Models*. https://www.arxiv.org/pdf/2410.02637

<br>

`ViTime`

![figure2](/assets/img/ts/img770.png)

<br>

`ViTST`

![figure2](/assets/img/ts/img771.png)

<br>

`VisualTimeAnomaly`

![figure2](/assets/img/ts/img772.png)

![figure2](/assets/img/ts/img773.png)

<br>

`TAMA` (Time Series Anomaly Multimodal Analyzer)

![figure2](/assets/img/ts/img774.png)

<br>

### b) Heatmaps

- Visualize TS in a 2D space 
  - Colors = Represent magnitudes

- Specifically useful for modeling **LONG TS**
- Examples) [143,219]
  - [143] *VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters* https://www.arxiv.org/pdf/2408.17253
  - [219] *Deep video prediction for timeseries forecasting*

<br>

`VisionTS`

![figure2](/assets/img/ts/img775.png)

![figure2](/assets/img/ts/img776.png)

![figure2](/assets/img/ts/img777.png)

<br>

### c) Spectrogram

Time series can be decomposed into the spectrum of frequencies and represented as a spectrogram. Wavelet transforms are a popular choice of representation for both univariate [220] and multivariate [144] tasks. 

### d) Other methods

Zhiguang and Tim [231] use Gramian Angular Fields (GAF) [20] to represent time-series. which visualize long and short termdependenciesbetter.Recurrenceplots(RP)Eckmannetal. [47] areanotherwaytocaptureperiodicpatternsintime-seriesused by [89] for classification and [110] forecasting. Time-VLM [161] combines information from Fourier coefficients, cosine and sine periodicity into a heatmap which is fed into a VLM encoder.
