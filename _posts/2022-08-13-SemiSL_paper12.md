---
title: (paper) SSL12 - An Overview of Deep Semi-Supervised Learning
categories: [SSL]
tags: []
excerpt: 2020
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# An Overview of Deep Semi-Supervised Learning (2020)

<br>

## Contents

0. Abstract
1. 

<br>

# 0. Abstract

semi-supervised learning (SSL)

- overcome the need for large annotated datasets

<br>

This paper :

$\rightarrow$ provide a comprehensive overview of Deep SSL

<br>

# 1. Introduction

## (1) SSL

![figure2](/assets/img/semi/img26.png)

<br>

## (2) SSL Methods

can be divided into following categories

1. **Consistency Regularization**

   - small perturbation in input ... small change in output 

2. **Proxy-label Methods**

   - produce additional training examples

3. **Generative Models**

   - learned features on one task can be transferred to other downstream tasks

   - Generative models that generate images from $p(x)$ ,

     must learn transferable features to a supervised task $p(y \mid x)$ 

4. **Graph-based Methods**

5. **Entropy Minimization**

   - force to make confident predictions

<br>

based on 2 dominant learning paradigims

- (1) **transductive learning**
  - apply the trained classifier on the unlabeled instances observed at training time
  - does not generalize to unobserved instances
  - mainly used on graphs
- (2) **inductive learning**
  - more popular paradigm
  - capable of generalizing to unobserved instances at test time

<br>

## (3) Main Assumptions in SSL

( refer to https://seunghan96.github.io/ssl/SemiSL_intro/ )

<br>

## (4) Related Problems

### a) Active Learning

- provided with a large pool of **unlabeled data**

- aims to carefully **choose the examples to be labeled** to achieve a higher accuracy
- [ domain ] where data may be abundant, but labels are scarce

<br>

2 widely used selection criteria

- (1) informativeness 
  - how well an unlabeled instance helps reduce the uncertainty of a statistical model
- (2) representativeness
  - how well an instance helps represent the structure of input patterns

<br>

Active Learning & SSL : both aim to use a limited amount of data to improve model

<br>

### b) Transfer Learning & Domain Adaptation

Transfer learning (TL) :

- to improve a learner on target domain,  by transferring the knowledge learned from source domain
- ex) Domain Adaptation (DA)

<br>

Domain Adaptation (DA)

- Goal : train a learner capable of generalizing across different domains of different distributions,

  where..

  	- (1) much labeled data for source domain
   - (2) no/less/fully labeled data for target domain
     	- no : unsupervised DA
     	- less : semi-supervised DA 
     	- fully : supervised DA 

<br>

SSL & unsupervised DA

- (common) 
  - provided with labeled and unlabeled data
  - goal : learning a function capable of generalizing to the unlabeled/unseen data
- (difference)
  - SSL : both labeled & unlabeld come from SAME distn
  - unsupervised DA : both labeled & unlabeld come from DIFFERENT distn

<br>

### c) Weakly-supervised Learning

Goal : same as supervised learning

Difference : instead of GT label ... provided with weakly annotated examples

- ex) crowd workers, output of other classifiers....

<br>

Example )

- weakly-supervised semantic segmentation,
  - pixel-level labels are substituted for inexact annotations

$\rightarrow$ SSL can be used to enhance the performance

<br>

### d) Learning with Noisy labels

If the noise is significant..... can harm much!

$\rightarrow$ to overcome this, s seek to correct the loss function!

<br>

Type of correction :

- [ex 1] SAME weight for all samples
  - relabel the noisy examples ( where proxy labels methods can be used )
- [ex 2] DIFFERENT ~
  - reweighing to the training examples to distinguish between the clean and noisy

<br>

SSL & Noisy Labels

- noisy examples are considered as unlabeled data 

  & used to regularize training using SSL methods

<br>

# 2. Consistency Regularization



