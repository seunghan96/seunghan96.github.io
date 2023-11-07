---
title: SpecSwap; A Simple Data Augmentation Method for End-to-End Speech Recognition
categories: [AUDIO, TS, CL]
tags: []
excerpt: Interspeech 2020
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# SpecSwap: A Simple Data Augmentation Method for End-to-End Speech Recognition (Interspeech, 2020)

https://www.isca-speech.org/archive/pdfs/interspeech_2020/song20b_interspeech.pdf

<br>

# Contents

0. Abstract
0. SpecSwap

<br>

# Abstract

### SpecSwap

- a simple DA for ASR
- acts directly on the spectrogram of input utterances
- swapping blocks of frequency channels & time steps

Architecture: Transformer-based networks 

<br>

# SpecSwap

( Inspired by SpecAugment )

- also deforms data at spectrogram level
- consists of two kinds of deformations of the log-mel spectrogram
  - (1) time swapping 
  - (2) frequency swapping

swapping blocks comes from our previous work

- [15] : permutation strategy
  - by reconstructing frames from a permuted speech feature sequence
  - Limitations
    - (a) Need to modify the attention structure and constructing special attention mask
    - (b) Need to be fine-tuned after pre-training

<br>

Question) ***Can we untie the (a) model structure and (b) permutation strategy?***

Solution) ***by applying permutation directly on spectrogram***

( = swapping blocks of features either in time-domain or frequency-domain )

<br>

![figure2](/assets/img/audio/img195.png)
