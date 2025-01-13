---
title: Are Language Models Actually Useful for Time Series Forecasting?
categories: [TS, NLP, LLM]
tags: []
excerpt: NeurIPS 2024
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Are Language Models Actually Useful for Time Series Forecasting?

<br>

# Contents

0. Abstract

1. Introduction

2. Related Works

3. Experimental Setup

   

<br>

# 0. Abstract

LLMs are being applied to TS forecasting

Ablation studies with 3 LLM-based TS forecasting models

$$\rightarrow$$ ***Removing the LLM component*** ( or replacing with basic attention ) ***does not degrade performance***!

Even do not perform better than models trained ***from scratch***!

<br>

# 1. Introduction

Claim: ***Popular LLM-based time series forecasters perform the same or worse than basic LLM-free ablations, yet require orders of magnitude more compute!***

Experiment

- 3 LLM-based forecasting methods
- 8 standard benchmark datasets + 5 datasets from MONASH

<br>

# 2. Related Work

## (1) TSF using LLMs

Chang et al., [5]

- Finetuning the certain modules in GPT-2 
- To align pre-trained LLMs with TS data for forecasting tasks

Zhou et al. [49] 

- Similar finetuning method, “OneFitAll”
- TS forecasting with GPT-2. 

Jin et al. [14]

- Reprogramming method to align LLM’s Word Embedding with TS embeddings
- Good representation of TS data on LLaMA 

<br>

## (2) Encoders in LLM TS Models

LLM for text: need "**work tokens**" ( $$1 \times d$$ vectors )

LLM for TS: need "**TS tokens**" ( $$1 \times d$$ vectors )

<br>

## (3) Small & Efficient Neural Forecasters

DLinear, FITS ....

<br>

# 3. Experimental Setup

## (1) Reference Methods for LLM and TS

![figure2](/assets/img/ts2/img225.png)

<br>

## (2) Proposed Ablations

![figure2](/assets/img/ts2/img226.png)

<br>

## (3) Datasets

![figure2](/assets/img/ts2/img224.png)
