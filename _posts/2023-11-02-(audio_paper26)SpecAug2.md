---
title: SpecAugment++; A Hidden Space Data Augmentation Method for Acoustic Scene Classification
categories: [AUDIO, TS, CL]
tags: []
excerpt: Interspeech 2021
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# SpecAugment++: A Hidden Space Data Augmentation Method for Acoustic Scene Classification (Interspeech, 2021)

https://arxiv.org/pdf/2103.16858.pdf

<br>

# Contents

0. Abstract
0. 

<br>

# Abstract

SpecAugment++

- a novel DA for acoustic scene classification (ASC)

- SpecAugment, mixup : only work on the input space

-  SpecAugment++ : applied to both the **input space and the hidden space**

  - **hidden state)** consist of ... 
    - masking blocks of frequency channels
    - masking blocks of time frames

- Imputing masked values

  - previous) zero ***( = ZM )***

  - proposed) 2 approaches 

    - based on the use of other samples within the minibatch

      - a) mini-batch based mixture masking ***( = MM )***
      - b) mini-batch based cutting masking ***( = CM )***

    - can be seen as introducing additional noises generated from the dataset

      & guide the networks to be more discriminative for classification

- Experimental results

  - DCASE 2018 Task1 dataset ... 3.6% gain
  - DCASE 2019 Task1 dataset ... 4.7% gain

<br>

# 1. Introduction

Different from the mixup [15] and BC learning [17] ...

$$\rightarrow$$ Labels of the augmented data are not changed!

<br>

# 2. SpecAugment++

![figure2](/assets/img/audio/img199.png)

<br>

![figure2](/assets/img/audio/img196.png)

![figure2](/assets/img/audio/img197.png)

![figure2](/assets/img/audio/img198.png)

<br>

# 3. Experiments

![figure2](/assets/img/audio/img200.png)
