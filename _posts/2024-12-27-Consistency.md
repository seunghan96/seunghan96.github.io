---
title: Consistency Models – Optimizing Diffusion Models Inference
categories: [LLM, NLP, CV, MULT, TS]
tags: []
excerpt: ICML 2023
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Consistency Models – Optimizing Diffusion Models Inference

```
Song, Yang, et al. "Consistency models." arXiv preprint arXiv:2303.01469 (2023).
```

참고: 

- https://aipapersacademy.com/consistency-models/
- https://arxiv.org/pdf/2303.01469

<br>

### Contents

1. Introduction
1. Consistency Models
1. Methods of Creating Consistency Models
1. Training Process

<br>

# 1. Introduction

**Consistency models** ( by OpenAI )

- New type of generative models 

<br>

Background: **Diffusion models**

- Backbone for dominant **image generation models **

  - (e.g., DALL-E 2, Imagen and Stable Diffusion) 

- Drawbacks: **# of inference steps**

  $$\rightarrow$$ Can be tooo slow!

![figure2](/assets/img/llm/img235.png)

<br>

# 2. Consistency Models

***Consistency models learn to remove all of the noise in one iteration***

![figure2](/assets/img/llm/img236.png)

<br>

### Consistency Models ( vs. Diffusion models )

Similar) Also learn to remove noise from an image

Difference) ***Dramatically reduce the number of steps***

- How? Teach the model to ***map*** between ***any image*** on the ***same denoisening path*** to the clear image

<br>

### (Term) ***Consistency***

The model learns to be

- **"consistent"** for producing the **"same" clear image** 
- for **any point** on the **same path**!

<br>

$$\rightarrow$$ Able to **jump directly** from a completely **noisy image** to a clear image

( + Flexibility: If higher quality image is needed, pay off with more compute! )

<br>

# 3. Methods of Creating Consistency Models

![figure2](/assets/img/llm/img237.png)

2 ways to train a consistency model

- (1) **Distillation** of a pre-trained diffusion model. 
  - (From) Large pre-trained diffusion model (e.g., Stable Diffusion (SD))
  - (To) Smaller consistency model

- (2) **From Scratch**
  - Able to produce good with trained-from-scratch models as well!

<br>

# 4. Training Process

![figure2](/assets/img/llm/img238.png)

Look at **"pairs of points"**

Feed the consistency model with **each** of the images from that **pair** to restore the original image

$$\rightarrow$$ Make these two results similar! (Minimize the difference)
