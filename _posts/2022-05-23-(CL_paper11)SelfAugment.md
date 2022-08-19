---
title: (paper 11) SelfAugment
categories: [CL]
tags: []
excerpt: 2021
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# SelfAugment : Automatic Augmentation Policies for Self-Supervised Learning

<br>

## Contents

0. Abstract
1. Introduction
2. Background & Related Work
   1. Self-supervised RL
   2. Learning data augmentation policies

3. Self-supervised Evaluation & Data Augmentation
   1. Self-supervised evaluation
   2. Self-supervised DA policies


<br>

# 0. Abstract

(common practice) Unsupervised Representation Learning

$$\rightarrow$$ use **labeled data** to evaluate the quality of representation

- this evaluation : guide line to **data augmentation policy**

<br>

But in real world …. **NO LABELS**!

<br>

This paper :

- evaluating the learned representation with a **SELF-supervised image rotation task** 

  is highly correlated with **SUPERVISED** evaluations 

  ( rank correlation > 0.94 )

- propose **Self-Augment**

<br>

### Self-Augment

**automatically & efficiently** select augmentation policies, **without using supervised evaluations**

<br>

# 1. Introduction

Recent works :

- used extensive **supervised evaluations** to choose **data augmentation policies**
- best policies = sweet spot
  - makes it **difficult to determine the corresponding image pair** , while retaining salient features

$$\rightarrow$$ in reality, hard to obtain labeled data!

<br>

Question :

***How to evaluate, w.o labeled data??***

<br>

### Contributions

1. Linear image-rotation-prediction evaluation task

   = highly correlated with downstream supervised task

2. Adapt 2 automatic Data Augmentation algorithms ( for instance CL )

3. Linear image-rotation-prediction :

   - works across network architectures
   - stronger CORR than jigsaw, color prediction task

<br>

Conclusion : ***IMAGE ROTATION PREDICTION*** is a strong & unsupervised evalaution criteria for **evaluating & selecting data augmentations** ( for instance CL )

<br>

# 2. Background & Related Work

## (1) Self-supervised RL

Common loss : **InfoNCE**

$$\mathcal{L}_{N C E}=-\mathbb{E}\left[\log \frac{\exp \left(\operatorname{sim}\left(\mathbf{z}_{1, \mathbf{i}}, \mathbf{z}_{2, \mathbf{i}}\right)\right.}{\sum_{j=1}^{K_{d}} \exp \left(\operatorname{sim}\left(\mathbf{z}_{1, \mathbf{i}}, \mathbf{z}_{2}, \mathbf{j}\right)\right)}\right]$$.

- has been shown to maximize a lower bound on the mutual information $$I\left(\mathbf{h}_{\mathbf{1}} ; \mathbf{h}_{\mathbf{2}}\right)$$

<br>

Algorithms :

- SimCLR : relies on **large batch size**

- MoCo : maintains a **large queue of contrasting images**

  $$\rightarrow$$ this paper focus on using MoCo for experiment!

<br>

Self-supervised model evaluation : done by..

- (1) separability
  - Network = frozen
  - Training data : trains a supervised linear model
- (2) transferability
  - Network = frozen / fine-tuned
  - Transfer Task Model ( fine-tuned using different dataset )
- (3) semi-supervised
  - Network = frozen / fine-tuned
  - either “separability” or “transferability” tasks

<br>

This paper seeks ***label-free & task-agnostic evaluation***

<br>

## (2) Learning data augmentation policies

this paper : use a ***self-supervised evaluation to automatically learn an augmentation policy for instance contrastive models***

<br>

FAA (Fast Auto Augment) 

- search based automatic augmentation framework

RandomAugment

- sampling-based approach

<br>

# 3. Self-supervised Evaluation & Data Augmentation

Central Goals

- (1) establish a **strong correlation** between SELF-supervised & SUPERVISED task

- (2) develop a practical algorithm for **SELF-supervised DA selection**

<br>

## (1) Self-supervised evaluation

LABELED data augmentation policy selection

- can directly optimize

UN-LABELED data augmentation policy selection

- seek an evaluation criteria, that is ***highly correlated with supervised performance without requiring labels***

<br>

Self-supervised tasks

- (1) rotation : $$\left\{0^{\circ}, 90^{\circ}, 180^{\circ}, 270^{\circ}\right\}$$
- (2) jigsaw : 4-way rotation prediction ( $$4 !=24$$ )
- (3) colorization 
  - input : grayscale image
  - output : pixel-wise classification ( on pre-defined color classes )

$$\rightarrow$$ These self-supervised tasks were originally used to learn representations themselves, but in this work, we evaluate the representations using these tasks.

<br>

## (2) Self-supervised DA policies

(1) Rand Augment ( = sampling-based strategy )

(2) Fast Auto Augment(FAA) ( = search-based strategy )

<br>

Notation

- each transformation : $$\mathcal{O}$$

  ```
  cutout, autoContrast, equalize, rotate, solarize, color, posterize,
  contrast, brightness, sharpnes, shear-x, shear-y, translate-x, translate-y, invert
  ```

- 2 parameters of $$\mathcal{O}$$ : 

  - (1) magnitude $$\lambda$$
  - (2) probability of applying the transformation $$p$$

- $$\mathcal{S}$$ : set of augmentation sub-policies

  - subpolicy $$\tau \in \mathcal{S}$$ : sequential application of $$N_{\tau}$$ consecutive transformation 

    ( $$\left\{\overline{\mathcal{O}}_{n}^{(\tau)}\left(x ; p_{n}^{(\tau)}, \lambda_{n}^{(\tau)}\right): n=1, \ldots, N_{\tau}\right\}$$ for $$n=1, \ldots, N_{\tau})$$  

    - each operation is applied with prob $$p$$

<br>

### a) SelfRandAugment

Assumption of **RandAugment**

1. all transformations share a ***single, discrete*** magnitude, $$\lambda \in[1,30]$$ 

2. all sub-policies apply the ***same number of transformations***, $$N_{\tau}$$ 

3. all transformations are applied with ***uniform probability***, $$p=K_{T}^{-1}$$ for the $$K_{T}= \mid \mathbb{O} \mid $$ transformations. 

$$\rightarrow$$ selects the best result from a grid seach over $$\left(N_{\tau}, \lambda\right)$$

<br>

Evaluate the searched $$\left(N_{\tau}, \lambda\right)$$ State, using a **self-supervised evaluation**

- (1) rotation, (2) jigsaw, (3) colorization

$$\rightarrow$$ ***SelfRandAugment***

<br>

### b) FAA algorithm<br>

Notation

- $$\mathcal{D}$$ : distribution on the data $$\mathcal{X}$$
- $$\mathcal{M}(\cdot \mid \theta): \mathcal{X}$$ : model
- $$\mathcal{L}(\theta \mid D)$$ : supervised loss

<br>

FAA (Fast AutoAugment)? 

- given pair of $$D_{\text {train }}$$ and $$D_{\text {valid }}$$, select augmentation policies that **approximately align the density of $$D_{\text {train }}$$ with the density of the augmented $$\mathcal{T}\left(D_{\text {valid }}\right)$$.**

- split $$D_{\text {train }}$$ into $$D_{\mathcal{A}}$$ , $$D_{\mathcal{M}}$$

  - train the model with $$D_{\mathcal{A}}$$

  - determine the policy with $$D_{\mathcal{M}}$$ 

    $$\rightarrow$$ $$\mathcal{T}^{*}=\underset{\mathcal{T}}{\operatorname{argmin}} \mathcal{L}\left(\theta_{\mathcal{M}} \mid \mathcal{T}\left(D_{\mathcal{A}}\right)\right)$$

- obtains final policy $$\mathcal{T}^{*}$$ by exploring $$B$$ candidate policies $$\mathcal{B}=\left\{\mathcal{T}_{1}, \ldots, \mathcal{T}_{B}\right\}$$ with BayesOpt

  - Samples a sequence of sub-policies from $$S$$

  - adjuts the …

    - (1) probabilities $$\left\{p_{1}, \ldots, p_{N_{\mathcal{T}}}\right\}$$ 
    - (2) magnitudes $$\left\{\lambda_{1}, \ldots, \lambda_{N_{\mathcal{T}}}\right\}$$

    to minimize $$\mathcal{L}(\theta \mid \cdot)$$ on $$\mathcal{T}\left(D_{\mathcal{A}}\right)$$ 

-  top $$P$$ policies from each data split are merged into $$\mathcal{T}^{*}$$

  $$\rightarrow$$ retrain using this policy on all training data to obtain the final network parameters $$\theta^{*}$$

<br>

### c) SelfAugment

SelfAugment = adapt the **search-based FAA algorithm**

<br>

[ 3 main differences from FAA ] 

- (1) Select the base policy

- (2) Search augmentation policies
- (3) Retrain MoCo using the full training dataset and augmentation policy

<br>

![figure2](/assets/img/cl/img33.png)
