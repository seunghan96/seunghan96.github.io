---
title: Speech LLMs; 3) Multimodal Information Fusion and Training Strategies
categories: [MULT, LLM, NLP, CV, AUDIO]
tags: []
excerpt: A Survey on Speech Large Language Models

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# A Survey on Speech LLMs

https://arxiv.org/pdf/2410.18908

<br>

## Contents

4.Multimodal Information Fusion

5.Training Strategies

<br>

# 4. Multimodal Information Fusion

**Critical issue** (in Speech LLM)??

$$\rightarrow$$ Alignment btw **(1) audio modality** & the **(2) text modality**

<br>

Requires 2 steps

- Step 1) **Audio Feature** Post-Process
- Step 2) **Audio** and **Text** Connection

<br>

## Step 1) "Audio Feature" Post-Process

$\rightarrow$ Focuses on determining **"what specific audio info"** is needed

<br>

Tend to directly use the ***"final layer" output*** of the encoder

Approaches

- \# 1. Extract the **"output of the final layer"** of the encoder 

- \# 2. Using "**intermediate layer outputs**" to capture more granular features
- \# 3. **"Attention"** mechanisms to emphasize relevant parts of the audio signal

<br>

## Step 2) Audio & Text Connection

$\rightarrow$ Addresses how to **"effectively combine"** these two types of information.

<br>

Audio feature must be integrated with the textual modality!

$$\rightarrow$$ To enable the **LLM** to perform the final inference. 

<br>

Classified into 2 categories:

- (1) Transforming the **audio feature** into the **textual** modality space
- (2) **Merging** the audio and textual modality spaces

<br>

### a) "Audio-to-Text" Modality Conversion

***LLMs are primarily designed for "TEXT"modalities!***

- How? Employ **"projector"** 
  - To transform the extracted audio modality features
- Effect? **Minimizes modifications to the LLM**

<br>

Two common methods are employed!

![figure2](/assets/img/llm/img535.png)

<br>

**(1) Direct Projection**

- Step 1) **Projection**
  - Directly projected into the **LLM’s text feature space**
- Step 2) Concatenate
  - Audio embeddings are then concatenated with the input text’s embedding vector
- Step 3) Feed to LLM

<br>

**(2) Token Mapping**

- Step 1) **Map into tokens**

  - Audio feature information is mapped into **text tokens**
  
- Step 2) Concatenate

  - Audio tokens are combined with the text tokens

    $$\rightarrow$$ Token sequence that includes both audio and text info

- Step 3) Feed to LLM

<br>

### b) "Combining" Audio and Text Feature Space

Above method: Does not achieve lossless modality fusion in the true sense

$$\rightarrow$$ **Information loss** may occur during modality conversion!

<br>

Solution: **Modify the original input space of the LLM** to integrate the audio modality

- Augments the token space by adding **audio tokens** on top of the **existing text tokens**, creating a new token space. 

![figure2](/assets/img/llm/img536.png)

<br>

# 5. Training Strategies

Training of current Speech LLMs: 3 approaches

- **(1) Pretraining**
- **(2) Supervised fine-tuning (SFT)**
- **(3) Reinforcement learning (RL)**

<br>

## (1) Pretraining

- Dataset: **Audio-text** pairs

- Common strategies: **SSL**
  - To better integrate **speech encoders with LLMs**, some researchers attempt to **re-pretrain speech encoders**

- Thorough re-training of multimodal large models is necessary!

<br>

## (2) Supervised Fine-Tuning (SFT)

$\rightarrow$ Further fine-tuning is often required!

<Br>

**Supervised fine-tuning**

- Common approach
- **Labeled data** from downstream task datasets is used to train the model
- To achieve alignment between the **(1) speech encoder** and the **(2) LLM**
  - To enhance **performance on specific tasks**
- Common training methods:
  - (1) Fine-tuning connectors
  - (2) Fine-tuning the encoder
  - (3) LLMs
- Involves handling **modality alignment** and completing the model’s learning of text-token mapping

<br>

## (3) Reinforcement Learning (RL)

Commonly used method in training LLMs

- Especially in the field of **safety alignment**

Ensures that the LLM optimizes in the desired direction!

 <br>

