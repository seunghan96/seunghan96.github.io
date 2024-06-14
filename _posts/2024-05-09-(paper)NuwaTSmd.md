---
title: NuwaTS; a Foundation Model Mending Every
Incomplete Time Series
categories: [TS]
tags: []
excerpt: arxiv
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# NuwaTS: a Foundation Model Mending Every Incomplete Time Series

<br>

![figure2](/assets/img/ts2/img81.png)

# Contents

0. Abstract

   


<br>

# 0. Abstract

NuwaTS

- Framework to repurpose **Pre-trained Language Model (PLM)** for general "**TS imputation**"
- Can be applied to incomplete TS **from any domain** with any missing patterns

<br>

**Specific embeddings** for each sub-series patch of the **incomplete TS**

These embeddings encapsulate information about ...

- (1) the **patch itself**
- (2) the **missing data patterns** within the patch
- (3) the patch’s **statistical characteristics**

<br>

Contrastive learning approach 

- to make representations of the **same patch more similar across different missing patterns**

<br>

Loss = (1) contrastive loss & (2) missing data imputation loss

$\rightarrow$ Train PLMs to obtain a **one-for-all imputation model**

( + Can be generalized to other TS tasks such as forecasting. )

<br>

One-for-all model 

- Impute incomplete TS data from any domain
- Accommodate any pattern of missing data

<br>



Effectively learning a **"one-for-all imputation model"** for diverse domains is challenging!

[Requirements]

- **strong adaptability** to various domains and missing data patterns.

- **capability to quickly specialize** to a specific domain with few-shot learning 

  ( + while retaining its generalizability )

<br>

### NuwaTS

**One-for-all model** for incomplete TS

Distinguished by several strategies:

- (1)  **Specific embeddings** 

  - information for the patch itself, 

    the missing data patterns within the patch, 

    and the patch’s statistical information 

  - becomes the input to PLMs

- (2) **Contrastive learning** 

  - encourages the model to produce more similar representations of the same patch under **varying missing data patterns**
  - Final loss = contrastive loss + reconstruction loss

- (3) Domain-specific prefix embedding 

  ( + plug-and-play fine-tuning mechanism )

  - Introduces modules that insert **well-designed continuous prompts** into each layer of the frozen pre-trained one-for-all model **without altering any of its weights**

<br>

# 2. Related Works

## (1) Incomplete TS Imputation

## (2) Foundation Models for TS



# 3. Methodology

