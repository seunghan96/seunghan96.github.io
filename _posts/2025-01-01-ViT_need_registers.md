---
title: Vision Transformers Need Registers
categories: [CV, MULT, DIFF]
tags: []
excerpt: ICLR 2024 Oral
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Vision Transformers Need Registers – Fixing a Bug in DINOv2?

```
Darcet, Timothée, et al. "Vision transformers need registers." ICLR 2024
```

참고: 

- https://aipapersacademy.com/vision-transformers-need-registers/
- https://arxiv.org/pdf/2309.16588

<br>

### Contents

1. Background: Visual Features
1. The Problem: Attention Map Artifacts
1. The Fix: Registers
1. Results
1. Conclusion

<br>

# Abstract

Proposal: **Vision Transformers (ViTs) registers**

- Share authors with DINOv2 paper

<br>

# 1. Background: Visual Features

![figure2](/assets/img/llm/img255.png)

- Models from scratch (X)

- Pre-trained large computer vision model (O)

  - e.g., DINOv2 ( a large Vision Transformer (ViT) model )

  - Output = **visual features** or **embeddings**

    $$\rightarrow$$ Capture the semantic of the input image

<bR>

# 2. The Problem: Attention Map Artifacts

## (1) Attention Map

ViT = Attention mechanism

- Attention map = Visualization of the attention values

  $$\rightarrow$$ Which parts of the image are important!

<br>

## (2) Object Discovery

Object Discovery: One usage for attention maps

- **Object detection** = Locates only known labeled object
- **Object discovery** = Also locates unknown not-labeled object
  - e.g.,) Object discovery using attention maps method = **LOST** (using DINOv1)

![figure2](/assets/img/llm/img256.png)

<br>

## (3) Artifacts

LOST method 

$$\rightarrow$$ Found that the attention map in DINOv2 is ***not as semantically clear*** as in DINOv1 !!

- Ex) Outlier peaks in DINOv2 attention map  = ***artifacts***

![figure2](/assets/img/llm/img257.png)

<br>

Not only DINOv2, but other large visual transformer models!

(e.g., OpenCLIP, DeIT)

![figure2](/assets/img/llm/img258.png)

<br>

## (4) Analyzing the Artifacts

**L2 norm values** of the features extracted for image patches 

- DINOv1: OK
- DINOv2: 
  - Majority of features are of low value 
  - But a small proportion of patches have high norm!

![figure2](/assets/img/llm/img259.png)

<br>

## (5) What Data Do The Artifacts Capture?

Conclusion

1. Artifacts lack **spatial information**
2. Artifacts hold **global information**

<br>

### a) Artifacts lack **spatial information**

**High-norm** features 

- Contain **less** information about their **position** in the original image

- (Left chart) **Orange line** 

  - **Artifacts** are located in patches that are **very similar** to their **surrounding patches**

    ( = Confirms that the artifacts **appear in background** )

- (Right chart) Trained models to ..

  - Task 1) predict the original position of a token

  - Task 2) reconstruct the original pixels

    $$\rightarrow$$ In both tasks, performs worse for the high-norm tokens!

![figure2](/assets/img/llm/img260.png)

<br>

### b) Artifacts hold **global information**

Classification results when using embeddings from DINOv2 as inputs

- (Row 1) class token
- **(Row 3) > (Row 2)**

![figure2](/assets/img/llm/img261.png)

<br>

## (6) When do the High-Norm Tokens Appear?

![figure2](/assets/img/llm/img262.png)

<br>

Figure 4(a)

- More **common** from the **middle to the last layers**

Figure 4(b)

- Start to appear **after training the model for a while**

  ( Not at the beginning of the training process )

Figure 4(c)

- Only appear on larger models

<br>

Conclusion:  **large and sufficiently trained models learn to recognize redundant tokens, and use them to store global information**. 

<br>

# 3. The Fix: Registers

## (1) Key Idea

Key idea) If the model learns to **use tokens** that are **less important** in order to store **global** information...

$$\rightarrow$$ We can ***add more tokens*** that the model will ***use to store that information***

( instead of the tokens from the original image )

<br>

## (2) Registers

![figure2](/assets/img/llm/img263.png)

<br>

Solution: Registers (= The added tokens)

- (1) **Added** to the **input**
- (2) **Discarded** from the **output**
  - Assumption: The model will use them instead of the image patch tokens to store the **global information**

<br>

## (3) Do Registers Prevent Artifacts?

![figure2](/assets/img/llm/img264.png)

![figure2](/assets/img/llm/img265.png)

<br>

# 4. Results

![figure2](/assets/img/llm/img266.png)

![figure2](/assets/img/llm/img267.png)

<br>

# 5. Conclusion

- Highlights how unimportant tokens store useful information.
- Registers nearly eliminate these artifacts.
- (Experiment 1) Classification, segmentation, and depth
  - Registers yield minor gains but increase memory and latency, making their use case-dependent.

- (Experiment 2) Object discovery 
  - Improves significantly with DINOv2 but remains inferior to DINOv1.
