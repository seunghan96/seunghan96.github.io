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



# 2. Recent Advances in Speech LLMs

## (1)  Evolution of Speech LLMs Architectures

Challenges in the field of SLU

- (1) **Long-form** speech understanding 
- (2) **Hotword** recognition

$\rightarrow$ Begun to explore the **integration of LLMs into SLU**

<br>

### a) Transformer Integration into Traditional Speech Models

[1] Dong et al. (2018) 

- **End-to-end** speech recognition system based on the **Transformer**

<br>

[2] Gulati et al. (2020) 

- Introduce **Conformer** 
  - Combines the local feature extraction capabilities of **CNN** with **Transformer**
- Make the Transformer more robust in processing speech signals

<br>

[3] **HuBERT** (Hidden-Unit BERT)

- Major breakthrough in utilizing **LLMs** to process audio features
- **SSL** on a large corpus of unlabeled speech data

<br>

[4] Radford et al. (2022) 

- **Whisper** model
- Integrated **multilingual and multitask** capabilities into a single model

<br>

[5] Junyi Ao et al. (2022)

- **SpeechT5**
- Unified encoder-decoder framework for various spoken language processing tasks

<br>

### b) Direct Audio Processing with Large Language Models

( Besides directly incorporating the LLMs into speech recognition tasks... )

Using LLM as a **whole to process audio features** extracted by traditional speech recognition tools!

- Leverages the powerful contextual understanding and reasoning abilities of LLMs
- To improve the accuracy and robustness of speech recognition and deeper speech understanding

$\rightarrow$ Gradually evolved into the ***main trend of using LLMs in the field of speech recognition***

= ***Speech LLMs***

<br>

**Aligning speech & text modalities** 

- by passing extracted features from speech to downstream language models

<br>

[1] Qi Liu et al.  (2020)

- First to explore this approach
- Separating the **mapping of speech feature information** from the **speech feature extraction process** into two independent modules 
- introducing the LSTM model into the process of mapping speech feature information to text [11]. 
- This approach differs from traditional CTC-based end-to-end systems by isolating the acoustic feature-to-text mapping as a standalone unit. 
- This separation laid the groundwork for the subsequent integration of large language models into this area, making it possible to embed LLMs more effectively in speech recognition tasks.



Speech LLMs can mainly be divided into 2 categories

- (1) Discrete Sequence Modeling
- (2) Continuous Sequence Modeling

<br>

(1) Discrete Sequence Modeling

- Condense audio feature information into discrete tokens
  - Tokens: Passed to the LLM for processing. 
- Example
  - Vall-E (2023)
    - LLMs to process audio features
    - Transformer architecture
      - Combine audio features with the capabilities of LLMs to achieve more natural text-to-speech generation
  - SpeechGPT 
    - Deep integration of speech models and LLMs
    - Not only of processing audio & but also interacting through natural language
    - New interactive paradigm to the field of speech recognition 
  - AudioPaLM (2023)
    - Expanded the capabilities of speech recognition into multimodal processing
    - Enhancing performance in multimodal tasks

<br>

(2) Continuous Sequence Modeling, 

- Audio feature information is projected into continuous input embedding vectors 
  - Vectors: Transmitted to the LLM for processing. 
- Example
  - Pengi (2023)
    - Projects speech modality information into the text space of LLMs without altering any parameters in LLM 
    - Result: Continuous audio feature information began to be used to convey richer representations to the LLM
  - SLAM-LLM (2024)
    - Addition of a linear projector
    - Allowing the task to be greatly completed by training only the projection layer

<br>

Address the issues inherent in these conventional methods!

[1]  Fathullah et al. (2024)

- Introduced the Conformer mechanism
- Achieving advancements in handling long speech sequences 

<br>

[2] SpeechX 

- Achieved breakthroughs in multilingual, multitask speech recognition
- Enabling seamless switching between languages and supporting challenges 
  - e.g., Long-form speech understanding and hotword recognition 

<br>

## (2) Advancements of Speech LLMs in Key Tasks and Challenges

Speech LLM paradigm 

- Significant success across various tasks

<br>

### a) Improvement in Traditional Tasks in SLU

Traditional tasks (in speech understanding)

- (1) Automatic speech recognition
- (2) Speaker identification
- (3) Speech translation

<br>

(1) Automatic Speech Recognition (ASR) 

- Task: Convert spoken language into text
- Modern ASR systems: Enhanced by LLMs!
- Aim to achieve ...
  - a) higher accuracy
  - b) better noise resilience
  - c) greater adaptability to diverse accents and dialects 

- Foundational for ...

  - a) voice-controlled applications
  - b) interactive voice response systems
  - c) automated transcription services. 

- Metric: Word Error Rate (WER).

- Traditional models:

  - Based on LSTM or GRU

  $\rightarrow$ Introduction of LLMs has significantly improved these results!

- In multilingual speech recognition, LLMs have demonstrated superior performance across various languages!
  - Dataset: Multilingual LibriSpeech (MLS) 

<br>

(2) Speech translation 

- Task: Converting spoken language from one language into written or spoken text in another language

- Involves two key steps

  - Step 1) Automatic speech recognition (ASR)
    - Transcribes spoken words into text
  - Step 2) Machine translation (MT)
    - Translates the transcribed text into the target language

- Used in real-time applications 

  - e.g., multilingual meetings, conferences, and live broadcasts

- With the successful application of LLMs in the field of MT

  $\rightarrow$ speech translation domain has also begun to gradually incorporate LLMs!

  

- Advancements
  - Not only has it improved the accuracy of speech translation tasks 
  - but it has also broadened the range of supported languages 

<br>

(3) Others

- LLMs have excelled in multitask learning scenarios!
- Example
  - Qwen-Audio model
    - Impressive performance in tasks that combine speech-to-text with other modalities
      - e.g., sentiment analysis and speaker identification
    - Reducing WER by 10% and improving sentiment recognition accuracy by 8% compared to single-task models 
