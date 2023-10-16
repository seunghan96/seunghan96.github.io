---
title: Tabular Data: Deep Learning is Not All You Need
categories: [TAB]
tags: []
excerpt: ICML 2021 workshop
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Tabular Data: Deep Learning is Not All You Need (ICML 2021)

https://arxiv.org/pdf/2106.03253.pdf

<br>

# Contents

0. Abstract

1. Introduction
2. DL for Tabular
   1. TabNet
   2. NODE
   3. DNF-Net
   4. 1D-CNN
   5. Ensemble of models
3. Comparing the models
   1. Experimental Setup
   2. Results

<br>

# Abstract

Exploere whether DL is needed in tabular!

Result

- (1) XGBoost outperforms DL models
- (2) XGBoost requires much less tunining
- (3) Ensemble of DL + XGBoost performs better than XGBoost alone

<br>

# 1. Introduction

DL in tabular

- usually use different datasets & no standard benchmark

$$\rightarrow$$ making it difficult to compare!

<br>

Questions.

- Q1) Are the models more accuracte for the unseen datasets ( not the datasets used in their papers )?
- Q2) How long does training & hyperparameter search takes?

<br>

Result: ***XGBoost is better & Ensemble of XGboost and DL is even better***

<br>

# 2. DL for Tabular

Algorithms

- TabNet
- NODE ( Neural Oblivious Decision Ensembles )
- DNF-Net
- 1D-CNN

<br>

## (1) TabNet

- sequencial decision steps encode features ( using sparse learned masks )
- select relevant features using masks ( with attention )
  - sparsemax layers: force to use small set of features

<br>

## (2) NODE ( Neural Oblivious Decision Ensembles )

- contains **equal-depth oblivious decision trees ( ODTs )**

  ( = ensemble of differentiable trees )

- only one feature is chosen at each level

  $$\rightarrow$$ balanced ODT

<br>

## (3) DNF-Net

- simulate **disjunctive normal formulas (DNF) in DNns**
- key = DNNF block
  - (1) FC layer
  - (2) DNNF layer ( formed by soft version of binary conjunctions over literals )

<br>

## (4) 1D-CNN

- best **single model** performance in Kaggle competetion with tabular data
- CNN : no local characterictics
- thus, use FC , then 1D-conv ( with short-cut connection )

<br>

## (5) Ensemble of models

includes 5 classifiers: (1)~(4) + XGbosot

- weight = normalized validation loss of each model

<br>

# 3. Comparing the models

Desirable properties

1. perform accuractly
2. efficient inference
3. short optimization time

<br>

## (1) Experimental Setup

### a) Datasets

4 DL models

11 datasets

- 9 datasets = 3 datasets x 3 papers
- 2 new unseen datasets ( from Kaggle )

<br>

### b) Optimization process

Bayesian optimization ( with HyperOpt )

Use 3 random seed initializations in the same partition & average performance

<br>

## (2) Results

### a) Do DL generalize well to other datasets?

1. DL models perform worse on **two unseen datasets**

2. XGBoost generally outperform DL

3. No DL models consistently outperform others

   ( 1D-CNN may seem to perform better )

4. Esnemble of DL & XGBoost outperforms others in mosts cases

<br>

![figure2](/assets/img/tab/img40.png)

<br>

### b) Do we need both XGBOost & DL?

3 types

- Simple ensemble: XGBoost + XVm + CatBoost
- Deep Ensembles w/o XGBoost: only (1) ~ (4)
- Deep Ensembles w/ XGBoost: (1) ~ (4) + XGBoost

<br>

### c) Subset of models

Ensemble improves accuracy!

But, additional computation...

$$\rightarrow$$ consider using **subsets of the models** within the ensemble

<br>

Criterion

- (1) validation loss ( model with low val error first )
- (2) based on model's uncertainty for each example 
- (3) random order

![figure2](/assets/img/tab/img41.png)

<br>

### d) How difficult is the optimization?

XGBoost outperformed the deep models, converging faster!

![figure2](/assets/img/tab/img42.png)

<br>

Results may be affected by several factors

1) Bayesian optimization hyperparams

2) Initial hyperparams of XGBoost may be more robust

   ( had previously been optimized over many datasets )

3) XGBoost's inherent characteristics.

<br>
