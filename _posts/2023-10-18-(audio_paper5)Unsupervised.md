---
title: Unsupervised feature learning for audio classification using convolutional deep belief networks
categories: [AUDIO, TS, CL]
tags: []
excerpt: NeurIPS 2009
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Unsupervised feature learning for audio classification using convolutional deep belief networks (NeurIPS 2009)

http://www.robotics.stanford.edu/~ang/papers/nips09-AudioConvolutionalDBN.pdf

<br>

# Contents

0. Abstract
0. Introduction
0. Unsupervised Feature Learning
0. Application to speech recognition tasks
0. Application to music classification tasks

<br>

# 0. Abstract

Apply convolutional deep belief networks to audio data 

& evaluate them on various audio classification tasks

<br>

# 1. Introduction

**DL** have not been extensively applied to auditory data. 

**Deep belief network**

- generative probabilistic model 
- composed of one **visible (observed) layer** and **many hidden layers**
- can be efficiently trained using greedy layerwise training

<br>

We will apply convolutional deep belief networks to **unlabeled auditory data**

$$\rightarrow$$ outperform other **baseline features (spectrogram and MFCC)**

<br>

Phone classification task

- MFCC features can be augmented with our features to improve accuracy

<br>

# 2. Unsupervised Feature Learning

## Training on unlabeled TIMIT data

TIMIT: large, unlabeled speech dataset

- step1) extract the spectrogram from each utterance of the TIMIT training data
  - spectrogram = 20 ms window size with 10 ms overlaps
  - spectrogram was further processed using PCA whitening (with 80 components) 
- step 2) train model

<br>

# 3. Application to speech recognition tasks

CDBN feature representations learned from the unlabeled speech corpus can be useful for multiple speech recognition tasks

- ex) speaker identification, gender classification, and phone classification

<br>

## (1) Speaker identification 

The subset of the TIMIT corpus

- 168 speakers and 10 utterances (sentences) per speake ( = total of 1680 utterances )

$$\rightarrow$$ 168-way classification 

<br>

Extracted a spectrogram from each utterance

- spectrogram = “RAW” features. 
- first and second-layer CDBN features using the spectrogram as input

![figure2](/assets/img/audio/img91.png)

<br>

## (2) Speaker gender classification

![figure2](/assets/img/audio/img92.png)

<br>

## (3) Phone classification

treat each phone segment as an individual example 

compute the spectrogram (RAW) and MFCC features for each phone segment. 

- 39 way phone classification accuracy on the test data for various numbers of training sentences

![figure2](/assets/img/audio/img93.png)

<br>

# 4. Application to music classification tasks

## (1) Music genre classification

Dataset

- unlabeled collection of music data.

- computed the spectrogram representation for individual songs
  - 20 ms window size with 10 ms overlaps)

- spectrogram was PCA-whitened 

<br>

Task: 5 way genre classification tasks: (classical, electric, jazz, pop, and rock) 

![figure2](/assets/img/audio/img94.png)

<br>

## (2) Music artist classification

4 way artist classification task

![figure2](/assets/img/audio/img95.png)
