---
title: (paper 65) SeqCLR
categories: [CL, TS]
tags: []
excerpt: 2020
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Contrastive Representation Learning for EEG Classification

<br>

## Contents

0. Abstract
1. Introduction
2. Method
   1. Channel Recombination & Preprocessing
   2. Channel Augmentations
   3. Learning Algorithm

<br>

# 0. Abstract

Interpreting and labeling EEG : challenging

<br>

### SeqCLR ( Sequential Contrastive Learning of Representations )

Present a framework for learning representations from EEG signals via contrastive learning

- recombine channels from multi-channel recordings

  $$\rightarrow$$ increase the number of samples quadratically per recording.

- train a channel-wise feature extractor 
  
  - by extending the SimCLR to TS
- introduce a set of augmentations for EEG

<br>

# 1. Introduction

Presenting a new framework that allows us to ..

- (1) combine multiple EEG datasets
- (2) use the underlying physics of EEG signals to multiply the number of samples
  (quadratic increase)
- (3) learn representations in a self-supervised manner via CL

<br>

Details

- Modify the SimCLR framework for TS
- In contrast to images, not clear what augmentations could be beneficial for TS
  - consulted EEG researchers to select a set of transformations

<br>

![figure2](/assets/img/cl/img165.png)

<br>

# 2. Method

## (1) Channel Recombination & Preprocessing

we obtain $$n \times (n-1) + n = n^2$$ new channels for $$n$$-channel recording

![figure2](/assets/img/cl/img164.png)

<br>

## (2) Channel Augmentations

A key ingredient of CL = augmentations

<br>

we chose the transformations as...

![figure2](/assets/img/cl/img166.png)

<br>

Strength of each transformation :

![figure2](/assets/img/cl/img167.png)

<br>

## (3) Learning Algorithm

SeqCLR ( Sequential Contrastive Learning of Representations )

- like SimCLR, contains 4 modules

<br>

![figure2](/assets/img/cl/img168.png)

<br>

1. Channel Augmenter

   - for each channel, the module **randomly applies 2 augmentations**
   - $$N \rightarrow 2N$$ augmented channels

2. Channel Encoder

   - transforms an input channel into **4 feature channels of same length**

   - enables us to encode **sequences of different lengths** for **different downstream tasks**

   - designed 2 encoder :

     - (1) A recurrent encoder 

       - with a multi-scale input  ( using down & up sampling of the channel ) 
       - uses 2 recurrent residual units 

     - (2) A convolutional encoder 

       - utilizes reflection paddings 

         ( to ensure the output signal is of the same length as the input signal )

       - uses 4 convolutional residual units

3. Projector

   - recurrent projection head

   - collapses the output of the encoder into 32-dim
   - uses downsampling & bidirectional LSTM units

4. Contrastive Loss

   - identical to NT-Xent ( in SimCLR )
   - $$\ell_{i, j}=-\log \frac{\exp \left(\operatorname{sim}\left(\boldsymbol{z}_i, \boldsymbol{z}_j\right) / \tau\right)}{\sum_{k \neq i}^{2 N} \exp \left(\operatorname{sim}\left(\boldsymbol{z}_i, \boldsymbol{z}_k\right) / \tau\right)}$$.

5. Classifier

   - ( for downstream cls task ) discard the projector & use classifier
   - details
     - output dim = \# of classes
     - log softmax