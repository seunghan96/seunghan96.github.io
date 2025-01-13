---
title: LLM-TS Integrator; Integrating LLM for Enhanced Time Series Modeling
categories: [TS, NLP, LLM]
tags: []
excerpt: NeurIPSW TSALM 2024
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# LLM-TS Integrator: Integrating LLM for Enhanced Time Series Modeling

<br>

# Previous Works

LLM for TS modeling

Previous works: Primiarility position LLM as the ***predictive backbone***

$$\rightarrow$$ Problematic!! omit the modeling within traditional TS models (e.g., periodicity)

<br>

## LLM-TS Integrator

Effectively integrates the capabilities of (1) LLMs into (2) traditional TS modeling

<br>

![figure2](/assets/img/ts2/img235.png)

<br>

Two modules

- (1) ***Mutual information (MI)*** module
  - Maximizing MI btw ..
    - (1) traditional model's representation
    - (2) LLM's textual representation
- (2) ***Sample reweighting*** module
  - Recognize that samples vary in importance for two losses
    - a) prediction loss
    - b) MI loss
  - Dynamically optimize these weight via bi-level optimization



