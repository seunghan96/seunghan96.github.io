---
title: (paper) Deep Semi-supervised Learning for Time Series Classification
categories: [SSL, TS]
tags: []
excerpt: 2022
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Deep Semi-supervised Learning for Time Series Classification (2022)

<br>

## Contents

0. Abstract
1. Introduction
2. From Images to Time Series
   1. Problem Formulation
   2. Backbone Architecture
   3. Data Augmentation
3. Methods
   1. Mean Teacher
   2. Virtual Adversarial Traning
   3. MixMatch
   4. Ladder Net

<br>

# 0. Abstract

Semi-supervised Learning

- mostly on CV ….. not on TS

<br>

This paper : discuss the **transferability** of SOTA models from **image to TS**

- model backbone
- data augmentation strategies

<br>

# 1. Introduciton

Question : ***Can we transfer SOTA SSL models from image to TS?***

- SOTA models : MixMatch, Virtual Adversarial Training, Mean Teacher, Ladder Net

<br>

2 modifications

- (1) modification of a suitable backbone architecture
- (2) adaptions of an appropriate DA strategy

<br>

propose four new deep SSL algorithms for TSC 

( + meaningful data augmentation )

<br>

# 2. From Images to Time Series

## (1) Problem Formulation

Time Series : $$\left\{\left\{x_{1,1}^{(i)}, \ldots, x_{1, t}^{(i)}\right\}, \ldots,\left\{x_{c, 1}^{(i)}, \ldots, x_{c, t}^{(i)}\right\}\right\}$$.

- $$t$$ : length
- $$c$$ : amount of covariates
  - $$c=1$$ : univariate
  - $$c>1$$ : multivariate
- $$x^{(i)} \in \mathcal{X} \subseteq \mathbb{R}^{c \times t}$$.

<br>

Input Space : $$\mathcal{X}$$

Target Space : $$\mathcal{Y}$$

- $$y^{(i)} \in \mathcal{Y}$$  …. categorical variable

<br>

Goal of SSL : train a prediction model $$f: \mathcal{X} \mapsto \mathcal{Y}$$ on a dataset $$\mathcal{D}=\left(\mathcal{D}^l, \mathcal{D}^u\right)$$

- labeled dataset : $$\mathcal{D}^l=\left\{\left(x^{(i)}, y^{(i)}\right)\right\}_{i=1}^{n_l}$$
- unlabeled dataset : $$\mathcal{D}^u=\left\{x^{(i)}\right\}_{i=n_l+1}^n$$

( where $$n=n_l+n_u$$  & $$n_l \ll n_u)$$ 

<br>

Batch of data : $$\mathcal{B} \subset \mathcal{D}$$ …… $$\mathcal{B}=\left(\mathcal{B}^l, \mathcal{B}^u\right)$$

- (labeled) $$\mathcal{B}^l \subseteq \mathcal{D}^l$$
- (unlabeled ) $$\mathcal{B}^u \subseteq \mathcal{D}^u$$

<br>

## (2) Backbone Architecture

Dimension

- Image : 3d tensor

- TS : 2d tensor ( channels : \# of covariates )

<br>

Fully Convolutional Network (FCN)

- use it as a backbone architecture
- outperforms a variety of models on 44 different TSC problems 

<br>

## (3) Data Augmentation

Regularization-based semisupervised methods

- injection of random noise into the model

<br>

DA strategies : $$g\left(x^{(i)}\right), g: \mathcal{X} \mapsto \mathcal{X}$$

- perturbate the input $$x^{(i)}$$ of a sample, 

  while preserving the meaning of its label $$y^{(i)}$$

<br>

propose the use of the **RandAugment strategy**

- removes the need for a separate search phase

<br>

( for each batch … )

$$\rightarrow$$ $$N$$ out of $$K$$ augmentation strategies are randomly chosen

( + **magnitude hyperparameter** is introduced to control the augmentation intensity )

<br>

Augmentation policies 

- warping in the time dimension
- warping the magnitude
- addition of Gaussian Noise
- random rescaling

<br>

# 3. Methods

## (1) Mean Teacher

- **consistency-regularization**-based models

- teacher model ( = average of the consecutive student models )

  $$\rightarrow$$ used to enforce consistency in model predictions

<br>

## (2) Virtual Adversarial Training (VAT) 

- **consistency-regularization**-based models

- a **small data perturbation** is learned

  ( maximum change in prediction )

- perturbed model predictions are used as **auxiliary labels**

<br>

## (3) MixMatch 

- various semi-supervised techniques  ( ex. consistency regularization, Mixup, pseudo-labeling ) are combined within one holistic approach

<br>

## (4) Ladder net

- reconstruction-based SSL model

  ( inspired by denoising autoencoders )

- extends a supervised encoder model with a corresponding decoder network

  ( able to use **unsupervised reconstruction loss** )

  

