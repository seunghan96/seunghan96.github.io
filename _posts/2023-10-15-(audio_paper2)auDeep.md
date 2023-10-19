---
title: auDeep: Unsupervised Learning of Representations from Audio with Deep RNNs
categories: [AUDIO, TS]
tags: []
excerpt: JMLR 2017
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# auDeep: Unsupervised Learning of Representations from Audio with Deep RNNs ( JMLR 2017 )

https://arxiv.org/pdf/1712.04382.pdf

<br>

# Contents

0. Abstract
0. Recurrent Seq2Seq AE
0. System Overivew
3. Experiments
   1. Three audio classification tasks
   2. Baselines
   3. Results


<br>

# Abstract

`auDeep`

- Python toolkit for **deep unsupervised representation learning from acoustic data**
- architecture
  - seq2seq ... consider temporal dynamics
- provide an extensive CLI in addition to a Python API 

code: https: //github.com/auDeep/auDeep. 

<br>

SOTA in audio classification

<br>

# 1. Recurrent Seq2Seq AE

Extends the seq2seq (RNN enc-dec model)

- input sequence : fed to a multi-layered RNN 
- final hidden state  fed to FC layer
- output : fed to decoder RNN & reconstruct the input sequence

Loss : RMSE (reconstruction loss) 

<br>

Details:

- for faster model convergence, the expected decoder output from the previous step is fed back as the input into the decoder RNN
- used representation : activations of the FC layer

<br>

Input data = **spectrograms**

- time dependent sequences of frequency vectors. 

<br>

Two of the key strengths

- (1) fully **UNsupervised** training
- (ii) the ability to account for the **temporal dynamics of sequences**

<br>

# 3. System Overview

![figure2](/assets/img/audio/img82.png)

- extracted features can be exported to CSV or ARFF for further processing

  ( ex. classification with alternate algorithms )

<br>

# 4. Experiments

## (1) Three audio classification tasks. 

- (1) Acoustic scene classification 
  - dataset : ( TUT Acoustic Scenes 2017 (TUT AS 2017) )
- (2) Environmental sound classification (ESC)
  - dataset : ( ESC-10 and ESC-50 )
- (3) Music genre classification 
  - dataset : ( GTZAN )

<br>

Train multiple autoencoder configurations using auDeep

& Perform **feature-level fusion** of the learned representations. 

$\rightarrow$ fused representations are evaluated using the built-in MLP with the same cross-validation setup as used for the baseline systems on the TUT AS 2017, ESC-10, and ESC-50 data sets. 

<br>

## (2) Baselines

(a) CNN (Piczak, 2015a) 

(b) Sparse coding approach ( Henaff et al., 2011 )

(c) SoundNet system (Aytar et al., 2016) 

- better than auDeep ....but not fair compairison
  - auDeep was trained using ESC-10 and ESC-50 data only 
  - SoundNet was pre-trained on an external corpus of 2+ million videos. 

<br>

## (3) Results

![figure2](/assets/img/audio/img83.png)
