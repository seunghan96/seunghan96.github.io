---
title: (VLM survey) (Part 6; Performance Comparison & Future Works)
categories: [MULT, LLM, NLP, CV]
tags: []
excerpt: arxiv 2024
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Vision-Language Models for Vision Tasks: A Survey

https://arxiv.org/pdf/2304.00685

<br>

![figure2](/assets/img/llm/img492.png)

# Contents



<br>

# Abstract

Integrate (1) & (2)

- (1) LLMs
- (2) **Spoken Language Understanding (SLU)**



Procedures

- Step 1) **Audio Feature Extraction** 
- Step 2) **Multimodal Information Fusion**
- Step 3) **LLM Inference** (Speech LLMs)

$\rightarrow$ Richer audio feature extraction + end-to-end fusion of audio and text modalities!

<br>

# 1. Introduction

## (1) LLMs

(1) LLMs: Excel in ...

- Parsing **contextually appropriate sentences** 
- Maintaining coherence over **multiple conversational turns**

$\rightarrow$ Crucial for tasks like **dialogue systems, automatic summarization, and machine translation**

<br>

(2) LLMs: Achieved remarkable success in **multimodal tasks**

- e.g., visual question answering, image generation

<br>

## (2) SLU

(3) **Spoken Language Understanding (SLU)**

= Interpreting spoken language 

- To extract meaning intent, and relevant information **beyond simple transcription**

<br>

(4) (Basics) **Speech-to-text tasks**

- e.g.,  automatic speech recognition (ASR)

<br>

(5) Modern systems: Adept at ...

- a) Handling **diverse accents and languages**

- b) Improving **efficiency and accuracy **in workflows 
  - e.g., medical transcription and customer service automation

<br>

## (3) Challengs of Speech LLM

(6) **Challenges of Speech LLM**

- a) **Long-form** recognition 

  - Struggles with maintaining context over **extended periods** 

    $\rightarrow$ Accuracy degradation & latency issues in real-time applications.

- b) **Hotword/keyword** recognition. 

  - Critical for wake word recognition ( e.g., Hey Siri~! )

  - Faces difficulties in **noisy environments** and balancing sensitivity and specificity, 

    ( especially when hotwords are contextually similar to other phrases. )

<br>

## (4) Contributions

1. Comprehensive survey analyzing **Speech LLMs in the SLU domain**.
   - Development of **Speech LLMs**
   - **Model architecture**
   - **Comparative analysis** ( vs. traditional speech models )

2. Training methods for **aligning speech & text modalities**
   - Emphasis on the potential development of RL (e.g., DPO, PPO)

3. Analyze the LLM’s Dormancy when the LLM is applied in the speech domain

   
