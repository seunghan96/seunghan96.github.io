---
title: One Fits All; Power General TS Analysis by Pretrained LM
categories: [TS,NLP,GAN]
tags: []
excerpt: NeurIPS 2023
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# One Fits All: Power General TS Analysis by Pretrained LM

<br>

# Contents

0. Abstract
0. Introduction
0. Related Works
   0. In-modality Transfer Learning
   0. Cross-modality Transfer Learning

0. Methodology
0. Experiments
0. Ablation Studies
0. GPT2 vs. BERT vs. BEiT
0. Efficiency Analysis

<br>

# Abstract

Main challenge of foundation model in TS = **Lack of large amount of TS data**

<br>

Solution) ***leverage CV or NLP model***

Frozen Pretrained Transformer (FPT)

- refrain from altering the self-attention & FFNN of residual blocks in pretrained NLP/CV model

<br>

![figure2](/assets/img/ts/img528.png)

<br>

# 1. Introduction

Advantage of foundation model

- provide a **unified framework for handling diverse tasks**
- ( $$\leftrightarrow$$ each task requires a specifically designed algorithm )

<br>

Problem in TS: ***lack of large data***

Solution: leverage **pre-trained language model**

- provide a **unified framework**
- self-attention modules in the pre-trained transformer acquire the ability to perform certain non-data-dependent operations through training

<br>

# 2. Related Works

## (1) In-modality Transfer Learning

Because of insufficient training sample, little research on pre-trained models

<br>

## (2) Cross-modality Transfer Learning

VLMo (2021)

- Stagewise pretraining strategy
- Utilize frozen attention blocks pretrained by **IMAGE** 
- Transfer to **LNAGUAGE**

<br>

Voice2series (2021)

- Leverage a pretrained **speech** processing model for **TS classification**

<br>

# 3. Methodology

( Focuse on GPT 2, but also experiment on BERT & BEiT )

![figure2](/assets/img/ts/img529.png)

<br>

# 4. Experiments

## (1) Imputation

Following **TimesNet**, use different random mask ratios (12.5, 25, 37.5, 50% )

![figure2](/assets/img/ts/img530.png)

<br.

## (2) Classification

10 multivariate UEA datasets

![figure2](/assets/img/ts/img531.png)

![figure2](/assets/img/ts/img532.png)

<br>

## (3) Anomaly Detection

5 commonly used datasets

- SMD, MSL, SMAP, SwaT, PSM

![figure2](/assets/img/ts/img533.png)

<br>

## (4) Long-term Forecasting

![figure2](/assets/img/ts/img534.png)

<br>

## (5) Short-term Forecasting

![figure2](/assets/img/ts/img535.png)

<br>

## (6) Few-shot Forecasting

![figure2](/assets/img/ts/img536.png)

<br>

## (7) Zero-shot forecasting

![figure2](/assets/img/ts/img537.png)

<br>

# 5. Ablation Studies

Several variants

- GPT2(0) FPT
- GPT2(6) w/o freezing
- GPT2(6) w/o pr-training

<br>

![figure2](/assets/img/ts/img538.png)

<br>

# 6. GPT2 vs. BERT vs. BEiT

![figure2](/assets/img/ts/img539.png)

<br>

# 7. Efficiency Analysis

![figure2](/assets/img/ts/img540.png)
