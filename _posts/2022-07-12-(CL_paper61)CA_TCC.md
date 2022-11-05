---
title: (paper 60) CA-TCC
categories: [CL, TS]
tags: []
excerpt: 2022
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Self-supervised Contrastive Representation Learning for Semi-supervised TSC

<br>

![figure2](/assets/img/cl/img125.png)

## Contents

0. Abstract
1. 

<br>

# 0. Abstract

propose a novel TS representation learning framework

- with TS-TCC ( Temporal and Contextual Contrasting )
- use contrastive learning

<br>

propose time-series specific **weak and strong augmentations**

& use their views to

- learn **robust temporal relations** in the proposed **temporal contrasting module**
- learn **discriminative representations** by our proposed **contextual contrasting module**

<br>

Details

- conduct a systematic study of **time-series data augmentation selection**

- extend TS-TCC to the semi-supervised learning settings

  $\rightarrow$ propose a Class-Aware TS-TCC (CA-TCC)

  - benefits from the available few labeled data
  - leverage robust pseudo labels produced by TS-TCC to realize class-aware contrastive loss. 

<br>

# 1. Introduction

Contrastive learning 

- strong ability over pretext tasks
- ability to learn invariant representations by contrasting different
  views of the input sample ( via augmentation )

<br>

image-based contrastive learning methods 

- may not able to work on TS

Why?

- reason 1) where its features are mostly spatial, we find
  time-series data are mainly characterised by the temporal
  dependencies 

- well on time-series data for the following reasons. First,
  unlike images, where its features are mostly spatial, we find
  time-series data are mainly characterised by the temporal
  dependencies [8]. Therefore, applying the aforementioned
  techniques directly to time-series data may not efficiently
  address the temporal features of data. Second, some augmentation
  techniques used for images such as color distortion,
  generally cannot fit well with time-series data. So
  far, few works on contrastive learning have been proposed
  for time-series data. For example, [9], [10] developed contrastive
  learning methods for bio-signals. However, these
  two methods are proposed for specific clinical applications
  and may not generalize to other time-series data.
  In this paper, we propose a novel framework that incorporates
  contrastive learning into self- and semi-supervised
  learning. Specifically, we propose a Time-Series representation
  learning framework via Temporal and Contextual
  Contrasting (TS-TCC) that is trained on totally unlabeled
  datasets. Our TS-TCC employs two contrastive learning and
  augmentation techniques to handle the temporal dependencies
  of time-series data. We propose simple yet efficient data
  augmentations that can fit any time-series data to create
  two different, but correlated views of the input samples.
  These views are then used by the two innovative contrastive
  learning modules. In the first module, we propose a novel
  temporal contrasting module to learn robust representations
  by designing a tough cross-view prediction task. Specifically,
  for a certain timestep, it utilizes the past latent features
  of one augmentation to predict the future of the other
  augmentation. This novel operation will force the model
  to learn robust representation by a harder prediction task
  against any perturbations introduced by different timesteps
  and augmentations. In the second module, we propose
  contextual contrasting to further learn discriminative representations
  upon the robust representations learned by the
  temporal contrasting module. In this contextual contrasting
  module, we aim to maximize the similarity among different
  contexts of the same sample while minimizing similarity
  among contexts of different samples. The pretrained model
  learns powerful representations about the time-series data
  regardless of downstream tasks.