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





### b) Long-Form Speech Understanding

**Traditional** speech recognition systems

$\rightarrow$ Struggled with long-form speech understanding!!

<br>

**Long-form** speech understanding

- Why? Due to ***context loss over extended periods***
- When? Particularly pronounced in ***audio segments longer than one minute***
  - **Traditional** models: Sharp increase in WER
  - **LLM-based** models: Significantly mitigated this problem!

<br>

**Whisper** (Audio)

- Maintains **contextual consistency** across long-form audio
- Results: Demonstrating an **18% reduction in WER** 
  - on audio segments exceeding five minutes 
  - compared to traditional models. 

<br>

**UniAudio & Pengi** (Speech)

- Remarkable performance in maintaining low WERs across extended speech segments 
- How? By integrating **advanced contextual understanding** 

<br>

### c) Hotword Recognition

Another challenging area for traditional speech recognition systems!

When? Especially in **noisy environments**

<br>

**GenTranslate model (2023)**

- Substantial improvements in this area!
- How? By leveraging the **contextual understanding capabilities of LLMs**
- Result:  22% improvement in hotword recognition accuracy (compared to traditional models)
- Others:
  - High robustness in noisy conditions

<br>

**Mala-ASR and Whisper**

- Not only improves hotword recognition accuracy
- But also  adapts dynamically to ***new hotwords in real-time***!

$\rightarrow$ Particularly valuable in ***dynamic environments*** 

(like live broadcasts or interactive voice response (IVR) systems)

<br>



### d) Real-time Multimodal Interaction

The integration of LLMs into speech recognition

$\rightarrow$ Expanded the scope of tasks beyond traditional speech-to-text

$\rightarrow$ Enabling **real-time multimodal interaction**

<br>

[1] VoxtLM and LauraGPT 

- Facilitate seamless integration of speech with visual and textual inputs
- Providing coherent and accurate multimodal outputs. 
- This capability is particularly valuable in applications such as live transcription and synchronized translation during presentations, where both speech and visual context need to be processed simultaneously.

<br>

LLM-based systems have introduced new functionalities!

$\rightarrow$ E.g., Generation of descriptive text, summaries, and even translations based on audio input. 



[2] ViOLA

- Generate coherent summaries and crosslanguage translations with high fluency and accuracy, 
- outperforming traditional models in both speed and quality of output.
- This represents a significant advancement in how speech recognition systems can interact with and interpret complex multimodal data streams [35].

<br>





