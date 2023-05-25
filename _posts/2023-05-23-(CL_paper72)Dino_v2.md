---
title: (paper 72) DINOv2; Learning Robust Visual Featurers without Supervision
categories: [CL, CV, TS]
tags: []
excerpt: 2023
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# DINOv2: Learning Robust Visual Features without Supervision

<br>

## Contents

0. Abstract
1. Introduction

<br>

# 0. Abstract

Revisit existing approaches & combine different techniques...

- to scale our pretraining in terms of data and model size

<br>

Most of the technical contributions aim at 

- accelerating and stabilizing the training at scale

<br>

(1) In terms of data

- propose an automatic pipeline to build a dedicated, diverse, and curated image dataset instead of uncurated data

(2) In terms of models

- train a ViT with 1B parameters and distill it into a series of smaller models

<br>

#  1. Introduction

[Paradigim in NLP]

Learning **task-agnostic pretrained representations** have become the standard in NLP

- use these features without fine-tuning
- better than those produced by task-specific models

$\rightarrow$ due to pretraining on large quantities of raw text using pretext objectives (w.o supervision)

<br>

How about in CV ?

- model that generate visual features that work out of the box on any task
  - image level ) image classification
  - pixel level ) segmentation

<br>

Promising efforts towards these foundation models

- (1) Text-guided pretraining

  - (limitation 1) limits the information that can be retained about the image 
    - since captions only approximate the rich information in images

  - (limitation 2) complex pixel-level information may not surface with this supervision.
  - (limitation 3) image encoders require aligned text-image corpora

- (2) self-supervised learning (SSL)

  - features are learned from images alone

  - (limitation 1) most of the advances in SSL were made in the context of pretraining on a small curated dataset (ImageNet-1k)

    - some were done on uncurated datasets

      $\rightarrow$ typically lead to a significant drop in the quality of the features

<br>

This paper explores ***if SSL has the potential to learn all-purposed visual features if pretrained on a large quantity of curated data.***

- Focus on stabilizing and accelerating discriminative self-supervised learning when scaling in model and data sizes.

<br>

(1) Pretraining dataset

- built an automatic pipeline to filter & rebalance datasets from uncurated images. 
  - inspired by pipelines used in NLP
    - where data similarities are used instead of external metadata & do not require manual annotation.

- major difficulty : rebalance concepts and avoid overfitting on a few dominant modes

  $\rightarrow$ naive clustering approach works reasonably well to resolve this issue.

<br>

(2) Pretrained models

- provide a variety of pretrained visual models = DINOv2 
  - trained with different ViT architectures

<br>

# 2. Related Work

## (1) Intra-Image SSL

Previous Works

- context Prediction ( Doersch et al. (2015) )

- re-colorizing images (Zhang et al., 2016)
- predicting transformations (Gidaris et al., 2018)
- inpainting (Pathak et al., 2016)
- patch re-ordering (Noroozi & Favaro, 2016; Misra & Maaten, 2020).

<br>

Recently, the emergence of patch-based architectures, like **ViTs**

$\rightarrow$ led to a revisit of **inpainting** for pre-training, potentially in feature space

<br>

MAE (Masked Auto-Encoder)

- learns features that provide substantial improvements when finetuned on downstream tasks
- This property of MAEs has been further validated on ..
  - video (Tong et al., 2022)
  - audio (Xu et al., 2022)
  - other modalities (Girdhar et al., 2022).

$\rightarrow$ ***However, their features require supervised finetuning, while our features perform well out of the box***

<br>

## (2) Discriminative SSL (Inter-Image SSL)

Main Idea : using discriminative signals between images ( or groups of images )

- became popular with the emergence of instance classification methods

- several improvements were made based either on instance-level objectives or clustering

Hard to scale to larger model sizes (Chen et al., 2021). 

<br>

## (3) Scaling SSL

Most of these works use large **uncurated data** to train models without supervision.

(1) Have shown that discriminative methods scale with data

- but because of the poor quality of the pretraining data ...

  $\rightarrow$ most of the results are obtained by finetuning the features

(2) Have also shown that these methods benefit from scaling in model size, given enough pretrained data

<br>

Questions the ability of SSL methods to work on ***any dat***a while we focus on producing the best pretrained encoders

<br>

## (4) Automatic data curation

