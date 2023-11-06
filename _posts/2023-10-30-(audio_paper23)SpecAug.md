---
title: SpecAugment; A Simple Data Augmentation Method for Automatic Speech Recognition
categories: [AUDIO, TS, CL]
tags: []
excerpt: arxiv 2019
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition (arxiv, 2019)

https://arxiv.org/pdf/2103.06695.pdf

<br>

# Contents

0. Abstract
0. Introduction
0. Augmentation Policy

<br>

# Abstract

SpecAugment,

- simple DA for speech recognition
- applied **directly to the feature inputs** of a NN **(i.e., filter bank coefficients)**
- consists of ..
  - warping the features
  - masking blocks of frequency channels
  - masking blocks of time steps

<br>

# 1. Introduction

Data augmentation for ASR

( ASR = Automatic Speech Recognition )

- Vocal Tract Length Normalization [11]
- Synthesize noisy audio [12]
- Speed perturbation for LVSCR tasks in [13]
- Use of an acoustic room simulator [14]
- Data augmentation for keyword spotting in [15, 16]
- Feature drop-outs for training multi-stream ASR systems [17]

<br>

### SpecAugment

- operates on the ***log mel spectrogram*** of the input audio

  ( rather than the raw audio itself )

- simple & computationally cheap

- consists of three kinds of deformations of the log mel spectrogram
  - (1) Time warping
    - a deformation of the TS in the time direction
  - (2) Time masking
  - (3) Frequency masking

<br>

# 2. Augmentation Policy

![figure2](/assets/img/audio/img189.png)

- ***individual*** augmentations applied to a single input
- log mel spectrograms are **normalized**  ( = zero mean )
  - $$\therefore$$ imputing masked values to zer o = setting it to mean value

<br>

![figure2](/assets/img/audio/img190.png)

- can apply ***multiple*** masks

