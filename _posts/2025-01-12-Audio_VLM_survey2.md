---
title: Speech LLMs; 2) Recent Advances in Speech LLMs
categories: [MULT, LLM, NLP, CV, AUDIO]
tags: []
excerpt: A Survey on Speech Large Language Models
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# A Survey on Speech LLMs

https://arxiv.org/pdf/2410.18908

<br>

# Contents



# 2. Recent Advances in Speech LLMs

## (1)  Evolution of Speech LLMs Architectures

Challenges in the field of SLU

- (1) **Long-form** speech understanding 
- (2) **Hotword** recognition

$\rightarrow$ Begun to explore the **integration of LLMs into SLU**

<br>

### a) Transformer + (Traditional) Speech Models

[1] **Speech-Transformer** (Dong et al. (2018)) 

( Title: *A no-recurrence sequence-to-sequence model for speech recognition* )

- **End-to-end** speech recognition system based on the **Transformer**

<br>

[2] **Conformer** (Gulati et al. (2020))

( Title: *Conformer: Convolution-augmented Transformer for Speech Recognition* )

- Combines **CNN** with **Transformer**
  - **Transformer**: Capture content-based "global" interactions
  - **CNN**: Exploit "local" features

- Make the Transformer more robust in processing speech signals!

![figure2](/assets/img/llm/img504.png)

<br>

[3] **HuBERT** (Hidden-Unit BERT)

( Title: *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units* )

- Major breakthrough in utilizing **LLMs** to process audio features
- **SSL** on a large corpus of unlabeled speech data
  - Offline clustering step to provide aligned target labels for a BERT-like prediction loss. 

- $L_m\left(f ; X,\left\{Z^{(k)}\right\}_k, M\right)=\sum_{t \in M} \sum_k \log p_f^{(k)}\left(z_t^{(k)} \mid \tilde{X}, t\right)$.
  - $p_f^{(k)}(c \mid \tilde{X}, t)=\frac{\exp \left(\operatorname{sim}\left(A^{(k)} o_t, e_c\right) / \tau\right)}{\sum_{c^{\prime}=1}^C \exp \left(\operatorname{sim}\left(A^{(k)} o_t, e_{c^{\prime}}\right) / \tau\right)}$.
  - (Output) feature sequence: $\left[o_1, \cdots, o_T\right]$.


![figure2](/assets/img/llm/img505.png)

<br>

[4] **Whisper** (Radford et al. (2022))

( Title: *Robust Speech Recognition via Large-Scale Weak Supervision* )

- Integrated **multilingual and multitask** capabilities into a single model
- Pretrained simply to predict **large amounts** of transcripts of audio on the internet
  - **680,000 hours** of multilingual and multitask supervision
- Also in **zero-shot** transfer setting

![figure2](/assets/img/llm/img506.png)

![figure2](/assets/img/llm/img507.png)

<br>

[5] **SpeechT5** (Junyi Ao et al. (2022))

( Title: *SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing* )

- Unified **encoder-decoder** framework 

  - For various spoken language processing tasks

- SpeechT5 consists of ...

  - (1) **Shared** encoder-decoder network 
  - (2) **Six modal-specific (speech/text)** pre/post-nets 

- Process

  - Step 1) Preprocess the input speech/text through the **pre-nets**
  - Step 2) Sequence-to-sequence transformation with shared **encoder-decoder network models** 
  - Step 3) Generate the output in the speech/text modality with **post-nets**
    - Based on the output of the decoder

- **Pre-training** (SSL) SpeechT5 to learn a unified-modal representation

  $\rightarrow$ Improve the modeling capability for both speech and text

- How to align the **textual and speech information**?

  $\rightarrow$ ***Cross-modal vector quantization*** approach 

  - Randomly mixes up speech/text states with latent units 

![figure2](/assets/img/llm/img508.png)

![figure2](/assets/img/llm/img509.png)

<br>

### b) Direct Audio Processing with LLMs

Using LLM as a **WHOLE to process audio features** (extracted by traditional speech recognition tools)

- Leverages the powerful **(1) contextual understanding** and **(2) reasoning abilities** of LLMs

$\rightarrow$ Gradually evolved into the ***main trend of using LLMs*** in the field of speech recognition

= ***Speech LLMs***

<br>

**Aligning speech & text modalities** 

- By passing **(extracted) features from speech** to **downstream language models**

<br>

[1] **Modular E2E ASR system** (Qi Liu et al.  (2020))

( Title: *Modular end-to-end automatic speech recognition framework for acoustic-to-word model* )

- First to explore this approach

  - Traditional CTC-based: Isolate the acoustic feature-to-text mapping as a **standalone unit**

- Separating the **mapping of speech feature information** from the **speech feature extraction process** into two independent modules:

  - **(1) Acoustic-tophoneme (A2P) model** 
    - a) Dataset: Trained on **"acoustic data"**
    - b) Task: Predicts the corresponding phoneme sequence with given acoustic features

  - **(2) Phoneme-to-word (P2W) model**
    - a) Dataset: Trained on both **"text and acoustic data"**
    - b) Task: Translates the phoneme sequence to the desired word sequence
    - Fine-tuned by the acoustic data

- **Decoding phase**: Two models will be integrated!

  $\rightarrow$  Act as a standard "**acoustic-to-word (A2W)**" model

- Can be easily trained with extra text data and decoded in the same way as a standard E2E ASR system

![figure2](/assets/img/llm/img510.png)

- **Phoneme synchronous decoding (PSD)**

  - Designed to **"speed up"** the decoding of CTC

  - **Removes the blank frames** in the phoneme posterior sequence

    $\rightarrow$ Greatly reduce the information rate without performance loss! 

  - This paper = **Downsampling layer** between the A2P and P2W network

<br>

Speech LLMs can mainly be divided into 2 categories

- (1) **Discrete** Sequence Modeling
- (2) **Continuous** Sequence Modeling

<br>

**(1) Discrete Sequence Modeling**

= Condense audio feature information into **discrete tokens**

<br>

Examples:

- [1] **Vall-E (2023)**

  ( Title: *Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers* )

  - LLMs to process audio features
  - Transformer architecture
    - Combine audio features with the capabilities of LLMs to achieve more natural text-to-speech generation
  - Pre-training: Scale up the TTS training data to 60K hours of English speech
  - Emerges in-context learning capabilities
  - Can be used to synthesize high-quality personalized speech 
    - with only a 3-second enrolled recording of an unseen speaker as an acoustic prompt!

![figure2](/assets/img/llm/img512.png)

![figure2](/assets/img/llm/img513.png)

<br>

![figure2](/assets/img/llm/img514.png)

- Dataset $\mathcal{D}=\left\{\mathbf{x}_i, \mathbf{y}_i\right\}$, 
  - $\mathbf{y}$ : audio sample 
  - $\mathbf{x}=\left\{x_0, x_1, \ldots, x_L\right\}$ : Corresponding phoneme transcription
  - (1) Pre-trained Neural codec model: $\operatorname{Encodec}(\mathbf{y})=\mathbf{C}^{T \times 8}$ 
    - To encode each audio sample into discrete acoustic codes
    - $\mathbf{C}$ = Two-dimensional acoustic code matrix
      - Row vector $\mathbf{c}_{t, \text { : }}$ : Eight codes for frame $t$ 
      - Column vector $\mathbf{c}_{:, j}$ : Code sequence from the $j$-th codebook, where $j \in\{1, \ldots, 8\}$. 
    - $T$ = Downsampled utterance length. 
  - (2) Neural codec decoder: $\operatorname{Decodec}(\mathbf{C}) \approx \hat{\mathbf{y}}$.
    - After quantization, reconstruct the waveform

<br>

- [2] **Vall-E 2 (2023)**

  ( Title: *VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers* )

  - Zero-shot text-to-speech synthesis (TTS)
  - Two enhancements (vs. Vall-E)
    - (1) Repetition Aware Sampling
    - (2) Grouped Code Modeling
  - **(1) Repetition Aware Sampling**
    - Refines the original nucleus sampling process 
      - By accounting for **token repetition in the decoding history**
    - Stabilizes the decoding & Circumvents the infinite loop issue
  - **(2) Grouped Code Modeling**
    - Organizes codec codes **into groups** 
      - To effectively shorten the sequence length
    - Boosts inference speed & Addresses the challenges of long sequence modeling

  ![figure2](/assets/img/llm/img511.png)

<br>

- [3] **SpeechGPT (2023)** 

  ( Title: *Speechgpt: Empowering large language models with intrinsic crossmodal conversational abilities* )

  - Trend: Multi-modal LLMs is crucial for AGI

  - Motivation: However, current speech LLMs typically adopt the cascade paradigm!

    $\rightarrow$ Preventing inter-modal knowledge transfer

  - Solution: SpeechGPT

    - Deep integration of speech models and LLMs

    - Not only of processing audio & but also interacting through natural language

    - New interactive paradigm to the field of speech recognition 

  - Dataset: Construct SpeechInstruct with discrete speech representations
    - Large-scale cross-modal speech instruction dataset. 
  - Three-stage training strategy
    - (1) Modality-adaptation pretraining
    - (2) Cross-modal instruction fine-tuning
    - (3) Chain-of-modality instruction fine-tuning
  - 논문 5쪽확인해보기

![figure2](/assets/img/llm/img515.png)

![figure2](/assets/img/llm/img516.png)

<br>

- [4] **AudioPaLM (2023)**
  - Expanded the capabilities of speech recognition into multimodal processing
  - Enhancing performance in multimodal tasks

<br>

**(2) Continuous Sequence Modeling** 

= Audio feature information is projected into **continuous input embedding vectors** 

<br>

Examples:

- Pengi (2023)
  - Projects speech modality information into the text space of LLMs without altering any parameters in LLM 
  - Result: Continuous audio feature information began to be used to convey richer representations to the LLM

<br>

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
