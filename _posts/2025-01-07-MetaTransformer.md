---
title: Meta-Transformer; A Unified Framework for Multimodal Learning
categories: [CV, MULT, LLM, NLP]
tags: []
excerpt: arxiv 2023
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Meta-Transformer: A Unified Framework for Multimodal Learning

```
Zhang, Yiyuan, et al. "Meta-transformer: A unified framework for multimodal learning." arXiv preprint arXiv:2307.10802 (2023).
```

참고: 

- https://aipapersacademy.com/meta-transformer/
- https://arxiv.org/pdf/2307.10802

<br>

### Contents

1. Introduction
2. Architecture
3. Unified Multimodal Transformer Pretraining
4. Experiments

<br>

# 1. Introduction

Meta-Transformer

- Multimodal learning
- Process information from **12 different modalities**

![figure2](/assets/img/llm/img302.png)

![figure2](/assets/img/llm/img303.png)

<br>

Challenges: ***Each data modality is structured differently***

<br>

# 2. Architecture

![figure2](/assets/img/llm/img304.png)

![figure2](/assets/img/llm/img306.png)

Goal: Produce embedding for any modality

- Can process different modalities as inputs

<br>

How can the transformer process information from different types of data? 

$$\rightarrow$$ **Data-to-sequence tokenizer**

- Consists of small tokenizers (per modality)

![figure2](/assets/img/llm/img307.png)

<br>

Task-specific head

- To solve various tasks with the obtained representation

<br>

# 3. Unified Multimodal Transformer Pretraining

( The paper does not share a lot of information about the pre-training process )

Dataset: **LAION-2B dataset**

- (Text,Image) pair dataset

Task: Contrastive learning

![figure2](/assets/img/llm/img305.png)

<br>

# 4. Experiments

![figure2](/assets/img/llm/img308.png)

<br>

## (1) Overall Performance

![figure2](/assets/img/llm/img309.png)

<br>

## (2) Text 

![figure2](/assets/img/llm/img310.png)

<br>

## (3) Image

![figure2](/assets/img/llm/img311.png)
