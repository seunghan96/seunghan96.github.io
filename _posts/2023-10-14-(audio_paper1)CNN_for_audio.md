---
title: CNN Architectures For Large-scale Audio Classification
categories: [AUDIO, TS]
tags: []
excerpt: ICASSP 2017
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# CNN Architectures For Large-scale Audio Classification (ICASSP 2017)

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7952132&tag=1

<br>

# Contents

0. Abstract
0. Introduction
0. Dataset
3. Experimental Framework
   1. Training
   2. Evaluation

4. Experiments

<br>

# Abstract

CNN architectures to classify the soundtracks of a dataset of $$70 \mathrm{M}$$ training videos ( 5.24 million hours) with 30,871 video-level labels. 

- DNNs
- AlexNet
- VGG
  INception
- ResNet

<br>

Experimenets on Audio Set ***Acoustic Event Detection (AED)*** classification task

<br>

# 1. Introduction

YouTube-100M dataset to investigate

- Q1) How popular **DNNs** compare on video soundtrack classification
- Q2) How performance varies with **different training set and label vocabulary sizes**
- Q3) Whether our trained models can also be useful **for AED**

<br>

### Conventional methods

Conventional AED

- Features : MFCCs 

- Classifiers : GMMs, HMMs, NMF, or SVMs

  ( recently; CNNs, RNNs )

<br>

Conventional datasets

- TRECVid [14], ActivityNet [15], Sports1M [16], and TUT/DCASE Acoustic scenes 2016 [17] 

  $$\rightarrow$$ much smaller than YouTube-100M. 

<br>

RNNs and CNNs have been used in Large Vocabulary Continuous Speech Recognition (LVCSR) 

$$\rightarrow$$ Labels apply to **entire videos** without any changes in time

<br>

# 2. Dataset

YouTube-100M data set += 100 million YouTube

- 70M training videos
- 20M validation videos
- 10M evaluation videos

<br>

Each video:

- avg) 4.6 minute $$\rightarrow$$ total 5.4M hourrs
- avg) 5 labels
  - labeled with 1 or more topic identifies ( among 30871 labels )
  - labels are assigned automatically based on a combination of metadata

Videos average 4.6 minute each for a total of 5.4M training hours

<br>

![figure2](/assets/img/audio/img77.png)

<br>

# 3. Experimental Framework

## (1) Training

### Framing

Audio : divided into non-overlapping 960 ms ***frames***

$$\rightarrow$$ 20 billion ***examples (frames)*** from the 70M videos

( inherits all the labels of its parent video 0

<br>

### Preprocessing to frames

Each frame is ...

- decomposed with a STFT applying 25ms windows evey 10 ms

- resulting spectrogram is integrated into 64 mel-spaced frequency bins

- magnitude of each bin is log-transformed

  ( + after adding a small offset to avoid numerical issues )

$$\rightarrow$$ RESULT:  ***log-mel spectrogram patches of 96 $$\times$$  64 bins*** ( = INPUT to cls )

<br>

Other details

- batch size = 128 (randomly from ALL patches)

- BN after all CNN layers

- final:  sigmoid layer ( $$\because$$ multi-**LAYER** classification )

- NO dropout, NO weight decay, NO regularization ...

  ( no overfitting due to 7M dataset )

- During training, we monitored progress via 1-best accuracy and mean Average Precision (mAP) over a validation subset.

<br>

## (2) Evaluation

10M evaluation videos

$$\rightarrow$$ create 3 balanced evaluation sets ( 33 examples per class )

- set 1) 1M videos ( 30K labels )
- set 2) 100K videos ( 3K labels )
- set 3) 12K videos ( for 400 most frequent labels )

<br>

Metric

- (1) balanced average across all classes of AUC
- (2) mean Average Precision (mAP)

<br>

# 3. Experiments

## (1) Arhictecture comparison

![figure2](/assets/img/audio/img78.png)

<br>

## (2) Label Set Size

![figure2](/assets/img/audio/img79.png)

<br>

## (3) Training Set Size

![figure2](/assets/img/audio/img80.png)

<br>

## (4) Qualitative Result

![figure2](/assets/img/audio/img81.png)
