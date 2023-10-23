---
title: Neural Acoustic Feature Extraction & Wav2vec
categories: [TS, AUDIO]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Neural Acoustic Feature Extraction

참고 : https://ratsgo.github.io/speechbook/docs/neuralfe

<br>

# 1. Introduction

NAF vs. MFCCs

- Neural Acoustic Feature Extraction : NN(learning)-based $$\rightarrow$$ stochastic

- MFCCs : rule-based $$\rightarrow$$ deterministic

<br>

Two algorithms:

- (1) Wave2Vec
  - key idea: Similarity ( 현재 음성 프레임, 다음 음성 프레임 ) $$\uparrow$$
- (2) SincNet
  - key idea: 새로운 CNN 계열 구조
- (3) PASE (Problem-Agnostic Speech Encoder)
  - based on Sincnet

<br>

# 2. Wav2Vec

### Architecture

( 둘 다 CNN 기반 )

- $$f$$ : encoder
- $$g$$ : context network

![figure2](/assets/img/audio/img73.png)

<br>

Task : binary classificaiton

- predict pos/negative pair 
  - positive = adjacent representation

<br>

# 3. VQ-Wav2Vec

Wav2Vec + **Vector Quantization**

![figure2](/assets/img/audio/img74.png)

<br>

## 방법 1) Vector Quantization

Vector Quantization with Gumbel Softmax

- step 1) calculate embedding $$\mathcal{Z}$$
- step 2) `linear` ($$\mathcal{Z}$$) ... logits
- step 3) `OHencode` (  `linear` ($$\mathcal{Z}$$) )

<br>

step 4) $$E \mathcal{Z}$$ ...... $$E$$: embedding matrix

![figure2](/assets/img/audio/img75.png)

<br>

## 방법 2) K-means Clustering

Distance btw $$E$$ and $$Z$$

$$\rightarrow$$ 가장 가까운 $$E$$ 벡터를 하나 선택

![figure2](/assets/img/audio/img76.png)
