---
title: Foundation Models for Time Series; A Survey
categories: [TS, LLM, MULT]
tags: []
excerpt: arxiv 2025
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Foundation Models for Time Series; A Survey

https://arxiv.org/pdf/2504.04011

Kottapalli, Siva Rama Krishna, et al. "Foundation Models for Time Series: A Survey." arXiv 2025

<br>

# Abstract

Foundation models for TS

- Architecture design

- Patch-based vs. Directly on raw TS
- Probabilistic vs. Deterministic
- Univariate TS vs. Multivariate TS
- Lightweight vs. Large-scale

- Type of objective function

<br>

# 1. Introduction

## (1) NN for TS Analysis

<br>

## (2) Transformer Paradigm

<br>

## (3) Transformer: Foundation Models for TS

<br>

# 2. Background

## (1) Unique Characteristics of TS

- Sequential Nature
- Temporal Dependencies
- Multivariate Complexities
- Irregular Sampling & Missing Data
- Noise & Non-Stationarity
- High Dimensionality in Long Sequences

<br>

## (2) Key Innovations of Transformers

### a) Attention mechanism & its role in Sequential data

Attention mechanisms provide the following advantages:

- (1) **Long-range dependency** modeling
- (2) **Dynamic** weighting
- (3) **Context-aware** representations

<br>

### b) Scalability & Parallelism

- (1) **Non-sequential** processing
- (2) **Efficient** handling of long-sequences
  - But $O(n^2)$ complexity $\rightarrow$ Sparse atetntion, linear Transformer ..

<br>

### c) Implication for TS modeling

The attention mechanism enables models to ...

- (1) Capture complex temporal dynamics
  - e.g., seasonality and long-term dependencies
- (2) Scalability ensures that these models remain practical for large-scale datasets

<br>

## (3) TS Applications

![figure2](/assets/img/ts/img726.png)

<br>

### a) TS Forecasting

<br>

### b) TS Imputation

- Transformer: Excel in learning contextual relationships to infer missing values
  - e.g., bidirectional attention, and encoder-decoder frameworks
- **TimeTransformer [80]**
  - Utilize self-attention mechanisms to predict missing data points in multidimensional datasets.

<br>

### c) Anomaly Detection

- Transformer: Powerful framework for anomaly detection due to their capacity for learning contextual representations 

- **Pretrained models**

  - Fine-tuned for anomaly detection tasks
  - By leveraging embeddings that capture normal behavior patterns

- **Transformer + VAE [84]**

- **Transformer + GAN [85]**

  $\rightarrow$ Further enhance AD by enabling **unsupervised** or **semi-supervised** learning

<br>

### d) TS Classification

<br>

### e) Change Point Detection

Task = Identifies moments when the statistical properties of a TS shift

- E.g., Detecting financial market shifts, climate pattern changes, and network traffic anomalies. 

<br>

### f) TS Clustering

<br>

## (4) FMs for TS

### a) Characteristics of FMs

Universal backbone for diverse downstream tasks

[Two-stage process]

- (1) Pretraining 
- (2) Fine-tuning

<br>

The ability of foundation models to generalize stems from several key properties:

- **(1) Task-agnostic pretraining objectives (SSL)**
  - NSP, MTM, CL .. 
- **(2) Scalability across domains**
  - Trained on heterogeneous datasets spanning multiple domains
  - Enhances their robustness and transferability to unseen tasks

- **(3) Adaptability through fine-tuning**

<br>

# 3. Taxonomy

## (1) Challenges in analyzing the field

<br>

## (2) Lack of Detailed Taxonomy

Key dimensions include ...

1. Model Architecture
   - (Transformer) Encoder-only, Decoder-only, Encoder-decoder
   - (Non-Transformer) e.g., **Tiny Time Mixers (TTM)**
2. Patch vs. Non-Patch
   - (Patch) Capture local temporal patterns before learning global dependencies
   - (Non-patch) Capture both short-and long-term dependencies across the full sequence
3. Objective Functions
   - (MSE) Regression tasks
   - (NLL) Probabilistic estimates that improve uncertainty modeling
4. UTS vs. MTS
5. Probabilistic vs. Non-probabilistic 
6. Model scale & complexity

<br>

# 4. Methodology

![figure2](/assets/img/ts/img727.png)

<br>

## (1) Model Architecture

### a) Non-Transformer

**Tiny Time Mixers (TTM)**

- Based on **TSMixer**
- Details
  - **a) Adaptive patching**: To handle multi-resolution
    - Different layers of the backbone operate at varying patch lengths
  - **b) Diverse resolution sampling**: To augment data to improve coverage across varying temporal resolutions
  - **c) Resolution prefix tuning**: To handle pretraining on varied dataset resolutions with minimal model capacity
  - **d) Multi-level modeling**: Capture channel correlations and infuse exogenous signals during fine-tuning. 
  - **e) Supports channel correlations and exogenous signals**

<br>

### b) Encoder-decoder

**TimeGPT**

- For TS forecasting
- Components from LLMs + CNN
- Details
  - [Transformer] Positional encoding & Multi-head attention
    - \+ Residual connections + LN
  - [CNN] For learning complex temporal patterns
  - [Dataset] Large, diverse time-series datasets
- Fine-tuned for specific forecasting tasks 
  - Using zero-shot or few-shot learning methods

![figure2](/assets/img/ts/img728.png)

<br>

### c) Encoder-only

**MOMENT** 

- Details
  - [Arch] Patching + Transformer + Relative PE + Instance norm
  - [SSL] MTM
  - [Dataset] Pretrained on a diverse collection of datasets ( Time Series Pile )
  - [Task] Forecasting, anomaly detection, and classification ...
- Key features
  - Handling variable-length TS
  - Scalability through a simple encoder and minimal parameters
  - Channel independence

![figure2](/assets/img/ts/img729.png)

![figure2](/assets/img/ts/img730.png)

<br>

**MOIRAI**

- Probabilistic MTS forecasting

- Handle data with varying frequencies and domains

- Details

  - [Arch] Patching + Transformer

    - Pre-normalization, RMSNorm, query-key normalization, and SwiGLU  .. 

  - [SSL] MTM

    - Trained with a CE loss 

      $\rightarrow$ Treating the forecasting task as a regression via classification

- Key features

  - Output = Mixture distribution
    - Capturing predictive uncertainty
    - Including Student’s t-distribution, negative binomial, log-normal, and low-variance normal distributions
  - Flexible patch size: To handle different frequencies (based on predefined size)
    - Larger patches for high-frequency data
    - Smaller ones for low-frequency data. 
  - Any-variate Attention mechanism
    - Flattens MTS into a single sequence

![figure2](/assets/img/ts/img731.png)

<br>

### d) Decoder-only

**Timer-XL**

- Key innovation: **TimeAttention** mechanism
  - Capture complex dependencies **within and across** TS
  - Incorporates both TD& CD via a **Kronecker product** approach
- Details
  - [SSL] NTP
  - [UTS & MTS]
    - For UTS
    - For MTS extends this approach by defining tokens for each variable and learning dependencies between them
  - Rotary Position Embeddings (RoPE) 
  - Capable of handling additional covariates

![figure2](/assets/img/ts/img732.png)

<br>

**Time-MOE**

- MoE + Decoder-only
  - MoE: Replace FFN with MoE layer
- Details
  - **Point-wise tokenization**: For efficient handling of variable-length sequences
    - \+ SwiGLU gating to embed time series points. 
  - Multi-resolution forecasting
    - Allowing predictions at multiple time scales (different forecasting horizons)

![figure2](/assets/img/ts/img733.png)

<br>

**Toto**

- For MTS forecasting
- Key innovations: Handle both TD & CD
- Details
  - [SSL] **NTP**
  - Probabilistic prediction head: Student-T Mixture Model (SMM)
    - Handle heavy-tailed distributions and outliers
  - Quantify uncertainty through Monte Carlo sampling

![figure2](/assets/img/ts/img734.png)

<br>

**Timer**

 applies large language models (LLMs) to time series forecasting by leveraging the sequential nature of both time series data and language. It emphasizes the use of extensive, high-quality datasets, such as the Unified Time Series Dataset [129] (UTSD) with up to 1 billion time points across seven domains, and integrates LOTSA [117] for zero-shot forecasting. Timer introduces a unified format called single-series sequence (S3) to handle diverse time series data, allowing for easier preprocessing and normalization without the need for alignment across domains. The model is trained using generative pre-training in an autoregressive manner, predicting future time series values based on historical context. A decoder-only Transformer architecture is employed to maintain the sequential dependencies inherent in time series data, making it well-suited for time series forecasting tasks. This design ensures scalable, adaptable, and effective forecasting across varied datasets.

<br>

**TimesFM**

a Transformer-based architecture designed for efficient long-horizon time-series forecasting. It operates by breaking input time-series into non-overlapping patches, which reduces computational costs and enhances inference speed. The model uses a decoder-only architecture, where it predicts the next patch based on prior patches, enabling parallel prediction and avoiding inefficiencies in multi-step autoregressive decoding. A random masking strategy is employed during training to handle variable context lengths, allowing the model to generalize across different input sizes. The core of the model consists of stacked Transformer layers with causal masking to ensure that future timesteps do not influence past predictions. During inference, the model generates future time-series in an auto-regressive manner, where each forecast is concatenated with the input for further prediction. The loss function used is Mean Squared Error (MSE), optimizing for point forecasting accuracy. TimesFM’s design offers flexibility in forecast horizons and context lengths, making it adaptable to various time-series datasets and suitable for zero-shot forecasting tasks

<br>

**Lag-LLaMA**

Transformer-based model for univariate probabilistic time series forecasting, built on the LLaMA [57] architecture. It incorporates a specialized tokenization scheme that includes lagged features (past values at specified lags) and temporal covariates (e.g., day-of-week, hour-of-day), allowing it to handle varying frequencies of time series data. The model uses a decoderonly Transformer with causal masking and Rotary Positional Encoding (RoPE) for sequential data processing. For forecasting, the output is passed through a distribution head that predicts the parameters of a probability distribution, providing not only point forecasts but also uncertainty quantification through probabilistic distributions. During training, it minimizes the negative log-likelihood of predicted distributions, and at inference, it generates multiple forecast trajectories through autoregressive decoding.

<br>

### e) Adapting LLM



## (2) Patch vs. Non-Patch

## (3) Objective Functions

## (4) UTS vs. MTS

## (5) Probabilistic vs. Non-probabilistic 

## (6) Model scale & complexity

