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
  - But $$O(n^2)$$ complexity $$\rightarrow$$ Sparse atetntion, linear Transformer ..

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

  $$\rightarrow$$ Further enhance AD by enabling **unsupervised** or **semi-supervised** learning

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

      $$\rightarrow$$ Treating the forecasting task as a regression via classification

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

(https://arxiv.org/pdf/2409.16040)

```
Xiang Shi, and others, Time-MoE: Billion-Scale Time Series Foundation Models
with Mixture of Experts, ICLR, 2025.
```

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

(https://arxiv.org/pdf/2407.07874)

```
Ben Cohen, Emaad Khwaja, Kan Wang, Charles Masson, Elise Ramé, Youssef
Doubli, and Othmane Abou-Amal, Toto: Time Series Optimized Transformer for
Observability, arXiv preprint arXiv:2407.07874, 2024, https://arxiv.org/abs/2407.
07874.
```

- For **MTS** forecasting $$\rightarrow$$ Handle both **TD & CD**
- Decoder-only model
- Details
  - [SSL] **NTP**
  - Probabilistic prediction head: Student-T Mixture Model (SMM)
    - Handle heavy-tailed distributions and outliers
  - Quantify uncertainty through Monte Carlo sampling

![figure2](/assets/img/ts/img734.png)

<br>

**Timer**

(https://arxiv.org/pdf/2402.02368)

```
Yuxuan Liu, Hao Zhang, Chenhan Li, Xiangyang Huang, Jiang Wang, and
Mingsheng Long, Timer: Generative Pre-Trained Transformers Are Large Time
Series Models, Forty-first International Conference on Machine Learning, 2024.
```

-  LLM to TS forecasting (Decoder-only model)
- Dataset 1: **Unified Time Series Dataset (UTSD)**
  - Up to 1 billion time points across seven domains
- Dataset 2: **Large-scale Open Time Series Archive (LOTSA)**
  - Over 27B observations across nine domains
  - For zero-shot forecasting
- **Single-series sequence (S3)**
  - Unified format to handle diverse time series data
  - For easier preprocessing and normalization w/o the need for alignment across domains
- Pretraining task
  - Decoder-only $$\rightarrow$$ Autoregressive Generative pre-training

<br>

Timer

![figure2](/assets/img/ts/img737.png)

<br>

UTSD

![figure2](/assets/img/ts/img735.png)

<br>

S3

![figure2](/assets/img/ts/img736.png)

<br>

**TimesFM**

(https://arxiv.org/pdf/2310.10688)

```
Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou, A decoder-only
foundation model for time-series forecasting, arXiv preprint arXiv:2310.10688, 2024,
https://arxiv.org/abs/2310.10688.
```

- Patchify TS
- Decoder-only architecture
- [SSL] Next Patch Prediction 
- Random masking strategy 
  - To handle variable context (input) lengths
- Summary: Flexibility in forecast horizons and context lengths

![figure2](/assets/img/ts/img739.png)

<br>

**Lag-LLaMA**

(https://arxiv.org/pdf/2310.08278)

```
Kashif Rasul, and others, Lag-Llama: Towards Foundation Models for Time Series
Forecasting, R0-FoMo:Robustness of Few-shot and Zero-shot Learning in Large
Foundation Models, 2023, https://openreview.net/forum?id=jYluzCLFDM.
```

- Univariate probabilistic TS forecasting

- Based on LLaMA

  - Decoder-only Transformer with causal masking
  - Rotary Positional Encoding (RoPE)

- Specialized tokenization scheme: Includes .. 

  - (1) Lagged features (past values at specified lags)
  - (2) Temporal covariates (e.g., day-of-week, hour-of-day)

  $$\rightarrow$$ Handle varying frequencies

- Probabilsitic forecasting

  - Output is passed through a distribution head 

    ( Predicts the parameters of a probability distribution )

- Loss function: NLL
- Inference: Multiple forecast trajectories through autoregressive decoding

![figure2](/assets/img/ts/img744.png)

<br>

### e) Adapting LLM

**Chronos**

(https://arxiv.org/pdf/2403.07815)

```
Ahmed F. Ansari, and others, Chronos: Learning the Language of Time Series,
arXiv preprint, 2024, https://arxiv.org/abs/2403.07815.
```

- Adapts LLM for probabilistic TS forecasting
- Novel tokenization approach
  - Continuous TS $$\rightarrow$$ Discrete tokens 
  - Step 1) Scaling the data (using mean normalization)
  - Step 2) Quantizing it through a binning process
    - Values are assigned to predefined bins
- Loss function: CE loss $$\rightarrow$$ Learn multimodal distributions
- Base model: 
  - T5 (encoder-decoder model)
  - (But can also be adapted to decoder-only models )
- Architecture remains largely unchanged from standard language models
- Minor adjustmnets
  - Vocabulary size to account for the quantization bins
- Pretraining task: Autoregressive probabilistic predictions

![figure2](/assets/img/ts/img743.png)

<br>

**AutoTimes** 

(https://arxiv.org/pdf/2402.02370)

```
Yuxuan Liu, Ganqu Qin, Xiangyang Huang, Jiang Wang, and Mingsheng Long,
AutoTimes: Autoregressive Time Series Forecasters via Large Language Models,
arXiv preprint arXiv:2402.02370, 2024.
```

- Adapts LLMs for MTS forecasting
- Patchify TS
  - Each segment = Single variate (treated independently)
- Timestamp position embeddings
- Pretraining task: Next token prediction
- Handle varying lookback & forecast lengths 
- (Summary) Key innovations 
  - Segment-wise tokenization
  - Timestamp embeddings for temporal context
  - Autoregressive multi-step forecasting

![figure2](/assets/img/ts/img740.png)

![figure2](/assets/img/ts/img741.png)

![figure2](/assets/img/ts/img742.png)

<br>

**LLMTime**

(https://arxiv.org/abs/2310.07820)

```
Nate Gruver, Marc Finzi, Shikai Qiu, and Andrew Gordon Wilson, Large
Language Models Are Zero-Shot Time Series Forecasters, NeurIPS 2023.
```

- Pretraining task: Next-token prediction
- TS = String of numerical digits
  - Each time step = Individual digits separated by spaces

![figure2](/assets/img/ts/img745.png)

![figure2](/assets/img/ts/img746.png)

<br>

**TIME-LLM**

(https://arxiv.org/pdf/2310.01728)

```
Mingyu Jin, and others, Time-LLM: Time Series Forecasting by Reprogramming
Large Language Models, ICLR 2024.
```

- Reprogramming framework

  - Adapts LLM to TS forecasting, w/o fine-tuning the backbone

- Transforming TS into text prototype representations

- Input TS : Before being reporgrammed with learned text prototypes...

  - Univarate TS + normalized, patched, embedded 

- Prompts: Augmented with domain-specific prompts

- Architecture

  - Frozen LLM
  - Only the input transformation and output projection parameters updated

  $$\rightarrow$$ Allow for efficient few-shot and zero-shot forecasting

![figure2](/assets/img/ts/img748.png)

![figure2](/assets/img/ts/img749.png)

![figure2](/assets/img/ts/img750.png)

![figure2](/assets/img/ts/img751.png)

<br>

**Frozen Pretrained Transformer (FPT)**

(https://arxiv.org/pdf/2103.05247)

```
Kevin Lu, Aditya Grover, Pieter Abbeel, and Igor Mordatch, Frozen Pretrained
Transformers as Universal Computation Engines, AAAI 2022
```

- Leverages pre-trained language or vision models
  - e.g., GPT [58], BERT [96], and BEiT [138]
- Freeze vs. Fine-tuning
  - [Freeze] Self-attention and FFN
  - [Fine-tune] Positional embedding, layer normalization, output layers
- Redesigned input embedding layer to project TS data into the required dimensions, employing linear probing to reduce training parameters

![figure2](/assets/img/ts/img747.png)

<br>

## (2) Patch vs. Non-Patch

### a) Patch-based

- Tiny Time Mixers

  - Non overlapping windows as patches during pre-training phase

- Timer-XL

  - Patch-level generation based on long-context sequences for MTS forecasting

- Toto 

  - Pre-trained on the next patch prediction

- MOMENT

  - Dividing TS into fixed-length segments, embedding each segment
  - Pretrain with MTM

- MOIRAI 

  - Patch-based approach to modeling time series with a masked encoder architecture

- AutoTimes

  - Each segment representing a single variate ( = Treated as individual tokens )
  - Capture inter-variate correlations while simplifying the temporal structure for the LLM

- Timer

  - TS is processed as single-series sequences (S3)

    = Each TS as a sequence of tokens

- TimesFM
  - Input TS is split into non-overlapping patches
- TIME-LLM 
  - Divite MTS into univariate patches
  - Reprogrammed with learned text prototypes
- Frozen Pretrained Transformer (FPT)
  - Patching

<br>

### b) Non Patch-based

Time-MOE

- Point-wise tokenization

TimeGPT

Chronos

- Discretizing the TS values into bins rather than splitting the data into fixed-size patches

Lag-LLaMA

- Does not use patching or segmentation 
- Rather, tokenizes TS data by incorporating lagged features and temporal covariates
  - Each token = Past values at specified lag indices + Additional time-based features

LLMTime

- TS as a string of numerical digits

  (  = Treating each time step as a sequence of tokens )

<br>

## (3) Objective Functions

### a) MSE

- Tiny Time Mixers
- Timer-XL
- MOMENT
- AutoTimes
- Timer
- TimesFM
- TIME-LLM
- Frozen Pretrained Transformer (FPT)

<br>

### b) Huber Loss

( by Time-MOE  )

<br>

Combines the advantages of MSE & MAE

$$L_{\delta}(r) = \begin{cases} \frac{1}{2} r^2 & \text{if} \  \mid r \mid  \leq \delta \\ \delta ( \mid r \mid  - \frac{1}{2} \delta) & \text{if} \  \mid r \mid  > \delta \end{cases}$$.

where:

- $$\delta > 0$$ is a user-defined threshold.
- If the residual is small ($$ \mid r \mid  \leq \delta$$), it behaves like MSE.
- If the residual is large ($$ \mid r \mid  > \delta$$), it behaves like MAE but transitions smoothly.

<br>

Summary

- For small errors, it uses the squared error (sensitive to small deviations).
- For large errors, it switches to absolute error (robust to outliers).

- Improve robustness to outliers and ensure stability during training

<br>

### c) LL & NLL

- Toto 
- Chronos
- Lag-LLaMA
- MOIRAI
- LLMTime (only training, no pretraining in LLMTime)

<br>

## (4) UTS vs. MTS

### a) Univariate

TimeGPT & Chronos & MOMENT & Lag-LLaMA

- Only UTS

<br>

### b) Multivariate

MOMENT & MOIRAI & Frozen Pretrained Transformer (FPT) & Tiny Time Mixers & Time-XL & Time-MOE & Toto & AutoTimes

- Both UTS & MTS

<br>

Timer 

- Primarily supports UTS
- But can treat MTS by flattening into single sequence! (feat. S3)

<br>

TimesFM 

- Appears to focus on UTS (no support for MTS)
- But still could theoretically accommodate MTS

<br>

## (5) Probabilistic vs. Non-probabilistic 

## (6) Model scale & complexity

