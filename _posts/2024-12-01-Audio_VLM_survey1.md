---
title: Speech LLMs; 1) Introduction
categories: [MULT, LLM, NLP, CV, AUDIO]
tags: []
excerpt: A Survey on Speech Large Language Models
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# A Survey on Speech LLMs

https://arxiv.org/pdf/2410.18908

<br>

## Contents

0. Abstract
1. Introduction
   1. LLMs
   2. SLU
   3. Challenges of Speech LLM
   4. Contributions

<br>

# Abstract

Integrate (1) & (2)

- (1) LLMs
- (2) **Spoken Language Understanding (SLU)**

<br>

Procedures

- Step 1) **Audio** Feature Extraction
- Step 2) **Multimodal** Information Fusion
- Step 3) **LLM Inference** (Speech LLMs)

<br>

Results

- (1) Richer **audio** feature extraction  
- (2) End-to-end fusion of **audio & text modalities**!

<br>

# 1. Introduction

## (1) LLMs

LLMs do well on ...

- Parsing **contextually appropriate sentences** 
- Maintaining coherence over **multiple conversational turns**

$$\rightarrow$$ Crucial for tasks like **"dialogue systems, automatic summarization, machine translation"**

<br>

Achieved remarkable success in **"multimodal"** tasks

- e.g., visual question answering, image generation

<br>

## (2) SLU

**Spoken Language Understanding (SLU)**

= Interpreting spoken language 

- To extract meaning intent, and relevant information **beyond simple transcription**
- Two steps
  - Step 1) **Automatic Speech Recognition (ASR)**
  - Step 2) **Natural Language Understanding (NLU)**


<br>

Modern systems: Adept at ...

- a) Handling **diverse accents & languages**

- b) Improving **efficiency & accuracy **in workflows 
  - e.g., Medical transcription and customer service automation

<br>

## (3) Challengs of Speech LLM

Challenge 1) "**Long-form** recognition" 

- Struggles with maintaining context over **extended periods** 

  $$\rightarrow$$ Accuracy degradation & latency issues in real-time applications.

<br>

Challenge 2) "**Hotword/keyword** recognition" 

- Critical for wake word recognition ( e.g., *Hey Siri~!* )

- Faces difficulties in **noisy environments** 

  ( = Balance btw **sensitivity & specificity** )

  - Especially when hotwords are **contextually similar to other phrases**

<br>

## (4) Contributions

1. Comprehensive survey analyzing **Speech LLMs** in the SLU domain
   - (1) Development of **Speech LLMs**
   - (2) **Model architecture**
   - (3) **Comparative analysis** ( vs. Traditional speech models )

2. Training methods for **aligning speech & text modalities**
   - Emphasis on the potential development of RL (e.g., DPO, PPO)

3. Analyze the LLM’s dormancy when the LLM is applied in the speech domain

   
