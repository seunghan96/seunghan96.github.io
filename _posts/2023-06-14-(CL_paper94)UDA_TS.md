---
title: (paper 94) Domain Adaptation for TS Under Feature and Label Shifts
categories: [TS,DA]
tags: []
excerpt: 2023
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Domain Adaptation for TS Under Feature and Label Shifts

<br>

Domain Adaptation for TS Under Feature and Label Shifts

## Contents

0. Abstract

1. Introduction

   

<br>

# 0. Abstract

Unsupervised domain adaptation (UDA) 

- labeled source domains
- unlabeled target domains

<br>

Transferring complex TS models : CHALLENGING !

(1) Dynamic temporal structure variations across domains. 

- **Feature shifts** in the time and frequency representations

(2) Difference in label distns ( in source & target )

<br>

### RAINCOAT

First model for both **closed-set** and **universal DA** on complex TS

- addresses feature & label shifts
- by considering both temporal and frequency features
  - align them across domains
  - correct for misalignments to facilitate the detection of private labels. 
- Improves transferability by identifying label shifts in target domains.

<br>

Experiments

- 5 datasets 
- 13 state-of-the-art UDA methods

<br>

# 1. Introduction

***Training models that can adapt to domain shifts is crucial!***

***Training a model that can detect unknown classes in test data is advantageous!***

<br>

DA is a highly complex problem

1. Must learn **highly generalizable features**
   - NN structure : poor ability to transfer across domains. 
2. Shifts in label distributions across domains may result in private labels
   - ex) class only in source domain
   - Unsupervised DA : must generalize across domains when labels from the target domain are not available during training

<br>

$\rightarrow$ Need for TS DA methods that ...

- (1) produce ***generalizable representations*** robust to feature and label shifts
- (2) expand the scope of existing DA methods by supporting ***both closed-set and universal DA***

<br>

More challenging when applied to TS

- (1) Can occur in both the time & frequency features of TS (Figure 1a)
- (2) Can fail to generalize due to shortcut learning (Brown et al., 2022)
  - occurs when the model focuses on time-space features while overlooking crucial underlying concepts in the frequency-space domain
- (3) Universal DA ( when no assumptions are made about the overlap between labels in the source & target domains ) is an unexplored area in TS ( Figure 1b )

<br>

### RAINCOAT 

**fRequency-augmented AlIgN-then-Correct for dOmain Adaptation for Time Series**

- a novel DA method for TS that handles both **feature and label shifts**
- first to address both **closed-set** & **universal DA** for TS
- Architecture 
  - **(1) Time-based encoder**
  - **(2) Frequency-based encoder**
- motivated by inductive bias that domain shifts can occur via both time or frequency feature shifts. 
- use Sinkhorn divergence for **source-target feature alignment**
- propose **"align-then-correct"** procedure for universal DA

<br>

# 2. Related Work

## (1) General Domain Adaptation

Three categories

1. Adversarial Training
2. Statistical Divergence
3. Self-Supervision

<br>

### a) Adversarial Training

Domain discriminator

- make features' domain indistinguishable

(Hoffman et al., 2015; Tzeng et al., 2017; Motiian et al., 2017; Long et al., 2018a; Hoffman et al., 2018). 

<br>

### b) Statistical Divergence: 

Extract domain invariant features 

- by minimizing domain discrepancy in a latent feature space. 

<br>

Examples

- MMD (Rozantsev et al., 2016)
- correlation alignment (CORAL) (Sun and Saenko, 2016)
- contrastive domain discrepancy (CDD) (Kang et al., 2019a)
- optimal transport distance (Courty et al., 2017; Redko et al., 2019)
- graph matching loss (Yan et al., 2016; Das and Lee, 2018). 

<br>

### c) Self-Supervision

Auxiliary self-supervision training tasks. 

Learn domain-invariant features through a pretext learning task

- ex)  data augmentation and reconstruction

Reconstruction-based methods : achieve alignment by ...

- (1) source domain classification 
- (2) reconstruction of target ( + source ) domain data 

(Ghifary et al., 2016; Jhuo et al., 2012)

<br>

## (2) DA for TS

### a) Adversarial Training

VRADA (Purushotham et al., 2017)

- builds upon a variational RNN
- trains adversarially to capture complex temporal relationships that are domain-invariant

<br>

CoDATS (Wilson et al., 2020 ) 

- VRADA + ( variational RNN -> CNN )

<br>

### b) Statistical divergence

SASA (Cai et al., 2021) 

- aligns the condition distribution of the TS data by minimizing the discrepancy of the associative structure of TS between domains

<br>

AdvSKM (Liu and Xue, 2021a) and (Ott et al., 2022) 

- metric-based methods
- align two domains by considering statistic divergence

<br>

### c)  Self-supervision

DAF (Jin et al., 2022) 

- extracts domain-invariant & domain-specific features
- perform forecasts for source and target domains through a shared attention module with a reconstruction task. 

<br>

CLUDA (Ozyurt et al., 2022) and CLADA (Wilson et al., 2021) 

- contrastive DA methods
- use augmentations to extract domain invariant and contextual features for prediction. 

<br>

[44,9,45,46,21,22]

### Common

Above methods align features without considering the potential gap between labels from both domains. Moreover, they focus on aligning only time features while ignoring the implicit frequency feature shift (Fig. 1 a). In contrast, RAINCOAT considers the frequency feature shift to mitigate both feature and label shift in DA.
Universal Domain Adaptation. Prevailing DA methods assume all labels in the target domain are also available in the source domain. This assumption, known as closed-set DA, posits that the domain gap is driven by feature shift (as opposed to label shift). However, the label overlap between the two domains is unknown in practice. Thus, assuming both feature and label shifts can cause the domain gap is more practical. In contrast to closed-set DA, universal domain adaptation (UniDA) (You et al., 2019) can account for label shift. UniDA categorizes target samples into common labels (present in both source and target domains) or private labels (present in the target domain only). UAN (You et al., 2019), CMU (Fu et al., 2020), and TNT (Chen et al., 2022a) use sample-level uncertainty criteria to measure domain transferability. Samples with lower uncertainty are prefer-
