---
title: Self-Rewarding Language Models
categories: [LLM, NLP]
tags: []
excerpt: ICML 2024
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Self-Rewarding Language Models

```
Yuan, Weizhe, et al. "Self-rewarding language models." ICML 2024
```

참고: 

- https://aipapersacademy.com/self-rewarding-language-models/
- https://arxiv.org/pdf/2401.10020

<br>

### Contents

1. Motivation
2. Background
   1. RLHF & RLAIF
   2. DPO

3. Self-Rewarding Language Models
   1. Key Idea
   2. Datasets for Training Self-rewarding LLMs
   3. Improving the Model Iteratively

4. Experiments

<br>

# 1. Motivation

Pre-trained LLMs

- Improved by getting **feedback** about the **model output** from humans
- Then, train on that feedback!

<br>

This paper

- In order to reach superhuman agents, future models require **superhuman feedback**!

<br>

### Previous works

There are already methods that ***use LLMs to be the ones the provide the feedback***

- e.g, **RLAIF (Reinforcement Learning from AI Feedback)** 

<br>

Then, what's novel with this work?

$$\rightarrow$$ The model providing the feedback for the model outputs is actually the **SAME model**!

<br>

# 2. Background

Previous works

- **Reinforcement learning from human feedback (RLHF)**

- **Reinforcement learning from AI feedback (RLAIF)**

<br>

## (1) RLHF & RLAIF

![figure2](/assets/img/llm/img153.png)

Procedure

- Step 1) **Pre-trained LLMs**: Generate responses

- Step 2) **Responses are evaluated **by human/AI (RLHF, RLAIF)

  $$\rightarrow$$ Produce a dataset of "**evaluated responses**"

- Step 3) Train a **reward model** based on that data
  - A model that learns to **rank the LLM responses**

- Step 4) LLM is trained using the **reward model**
  - Mostly via PPO (to yield outputs with high ranking)

<br>

## (2) DPO

![figure2](/assets/img/llm/img154.png)

Direct Preference Optimization (DPO)

- ***The need for the reward model is removed***

  $$\rightarrow$$ Train the LLM using the feedback data ***directly*** (w/o reward model)

<br>

# 3. Self-Rewarding Language Models

## (1) Key Idea

Self-rewarding LLM : should both learn to ..

- (1) Follow instructions
- (2) Act as a reward model

$$\rightarrow$$ That is, the LLM is used to generate responses and ***their*** evaluations,

( Keeps on training on these evaluated responses ***iteratively*** )

<br>

## (2) Datasets for Training Self-rewarding LLMs

![figure2](/assets/img/llm/img155.png)

Start with.. 

- (1) Pre-trained LLM = $$M_0$$
- (2) Two initial small datasets = **IFT + EFT**

<br>

### a) Instruction fine-tuning (IFT)

Instruction following dataset

- Crafted by **humans**

<br>

### b) Evaluation fine-tuning (EFT)

Samples that have an ...

- (1) Evaluation instruction prompt
  - Prompt that asks the model to evaluate the quality of a given response to a particular instruction. 

- (2) Evaluation result response
  - Determines the score of the response, with reasoning for that decision. 

$$\rightarrow$$ Serves as training data for the LLM to fill the **role of a reward model**

<br>

## (3) Improving the Model Iteratively

**Self-alignment** process that consist of **iterations**

- Each iteration has **two phases**

![figure2](/assets/img/llm/img156.png)

<br>

<br>

# 4. Experiments

![figure2](/assets/img/llm/img157.png)

![figure2](/assets/img/llm/img158.png)

$$\rightarrow$$ The more advanced versions win previous versions of the model!

