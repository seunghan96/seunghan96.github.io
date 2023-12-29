---
title: Rethinking CNN Models for Audio Classification
categories: [AUDIO, TS, CL]
tags: []
excerpt: arxiv 2020
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Rethinking CNN Models for Audio Classification ( arxiv 2020)

https://arxiv.org/pdf/2007.11154.pdf

<br>

![figure2](/assets/img/audio/img140.png)

# Contents

0. Abstract
1. Introduction
2. Related Work
   1. Audio Classification
   2. Transfer Learning for Audio Classification
   3. From IMage CLS to Audio CLS



<br>

# Abstract

Show that **ImageNet-Pretrained deep CNN models** can be used as strong baseline networks for **audio classification**

- although **significant difference** btw Spectrogram and ImageNet ...

  $$\rightarrow$$ transfer learning assumptions still hold firmly !!

<br>

To understand **what enables the ImageNet pretrained models to learn useful audio representations**, we systematically study **how much of pretrained weights is useful** for learning spectrograms. 

- (1) for a given standard model using pretrained weights is better than using randomly initialized weights 
- (2) qualitative results of what the CNNs learn from the spectrograms by visualizing the gradients. 

<br>

# 1. Introduction

**Spectrograms**

- have become increasingly popular in recent times **due to CNN**

- However, **natural images $$\neq$$ 2D spectogram**

  $$\because$$ spectrograms contain a **temporal dimension**

<br>

Modifications to the original CNN ( for spectogram )

- ex) kernels that move along in only one direction to capture **temporal data**
- ex) add **RNN structure** [8], [9], [10] or **Attention** [11], [12] or a combination of both CNN and RNNs [13], [14], [5] 
  - to improve the **sequential understanding** of the data.
- ex) [15] showed that we can treat these spectrograms as images and use the standard architecture like AlexNet [16] pretrained on ImageNet [17] for audio classification task.

<br>

Using pretrained ImageNet models

- has been no work that has used these **pretrained ImageNet models** for audio tasks
- most of the works shifted their focus to building models that were **more tailored for audio data**. 
  - ex) Models pretrained on large audio datasets like AudioSet[20] or the Million Songs Dataset[21] 

- people **have ignored a strong ImageNet pretrained model** baseline 

<br>

Proposal : show that by using **standard architectures** pretrained on ImageNet  and a **single set of input features like Melspectrograms**, we can achieve SOTA results on various datasets 

- like ESC-50 [25] , UrbanSound8k [26] and above 90% accuracy on the GTZAN dataset.

<br>

# 2. Related Work

## (1) Audio Classification

Variety of tasks

- Music Genre Classification[29], [30], [31]
- Environment Sound Classification\[32\], \[33\], \[34\] 
- Audio Generation[35], [36]. 

<br>

(1) Raw audio waveforms $$\rightarrow$$ use 1D-Conv

- ex) EnvNet [37], Sample-CNN [1]  : use raw audio as their input

<br>

(2) Most of the SOTA 

- use CNNs on Spectrograms
- use multiple models that take different inputs whose outputs are aggregated to make the predictions
  - ex) [18] used three networks to operate on the **(1) raw audio**, **(2) spectrograms**, and the **(3) delta STFT coefficients**
  - ex) [38] used two networks with **(1) mel-spectrograms** and **(2) MFCCs** as inputs to the two networks. 

<br>

This paper = show that with simple ***mel-spectrograms*** one can achieve SOTA!

<br>

## (2) Transfer Learning for Audio Classification

mainly focused on **pretraining a model on a large corpus of audio datasets like AudioSet, Million Songs Dataset.**

-  [45] : 
  - arch : a simple CNN network on the 
  - pretrain dataset : Million Song Dataset
  - downstream tasks : as Audio Event Classification, Emotion Prediction
- [46] :
  - arch : large scale models like VGG, Inception & ResNet
  - dataset : AudioSet
  - downstream tasks : for audio classification

<br>

This paper study transfer learning from **massive image datasets like ImageNet.**

<br>

## (3) From Image CLS to Audio CLS

[15] : one of the first papers to use models **pretrained on ImageNet** for audio classification.

[49], [50], [32] :  few works that use models **pretrained on ImageNet** for audio tasks

$$\rightarrow$$ these papers did not fully recognize the potential of these models since **they made several modifications to the design**

<br>

This paper = show that using a **single model** and a **single set of input features** we are able to achieve SOTA performance on a variety of tasks thereby reducing the time and space complexity of developing models for audio classification.
