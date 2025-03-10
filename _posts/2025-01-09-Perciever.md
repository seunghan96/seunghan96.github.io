---
title: Perceiver IO; A General Architecture for Structured Inputs & Outputs
categories: [LLM, CV, NLP, MULT]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Perceiver IO: A General Architecture for Structured Inputs & Outputs

참고: https://medium.com/analytics-vidhya/perceiver-io-a-general-architecture-for-structured-inputs-outputs-4ad669315e7f

<br>

### Contents

- Introduction
- (1) Perceiver
- (2) Perceiver IO

<br>

# Introduction

(1) Perceiver IO

- **Generalizable algorithm** that utilizes **transformer** that solves the **quadratic complexity**

- Extension of the original **perceiver** 

  $$\rightarrow$$ Extend to **any size of output values**

<br>

(2) Limitation of Transformer

- Quadratic complexity!

<br>

(3) Previous works

- [VIT Transformers](https://arxiv.org/abs/2010.11929),

  - Patchify the images & feed to transformer

  $$\rightarrow$$ Still doesn’t solve the quadratic complexity

<br>

# 1. (Original) Perceiver

(https://arxiv.org/abs/2103.03206)

Goal: Solve the **quadratic complexity** of Transformers

How? Add a **cross attention** layer between the ..

- (1) input sequence 
- (2) multi-headed attention

<br>

![figure2](/assets/img/llm/img62.png)

<br>

# 2. Perceiver IO

Add a **cross attention mechanism** in the **last layer of the decoder.**

$$\rightarrow$$ Maps latent of the encoder to **arbitrarily sized** and structured outputs using a **querying system** ( = simply **querying the latent array** using a **query feature vector** unique to the desired output element )

<br>

![figure2](/assets/img/llm/img63.png)
