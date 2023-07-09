---
title: (paper 93) Contrastive Learning for Unsupervised Domain Adaptation of Time Series
categories: [CL, TS]
tags: []
excerpt: 2023
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Contrastive Learning for Unsupervised Domain Adaptation of Time Series

<br>

## Contents

0. Abstract
1. Introduction
2. Related Works
   1. Unsupervised Domain Adaptation (UDA)
   2. UDA for TS

3. Problem Definition
4. Proposed CLUDA Framework
   1. Architecture
   2. Adversarial Training for UDA
   3. Capturing Contextual Representations
   4. Aligning the Contextual Representation Across Domains

5. Experimental Setups
6. Results

<br>

# 0. Abstract

Unsupervised domain adaptation (UDA) 

- learn model using **LABELED** source domain that performs well on **UNLABELED** target domain

<br>

### CLUDA

develop a novel framework for UDA of TS data

propose a CL framework in MTS

- preserve label information for the prediction task. 

capture the variation in the contextual representations between source and target domain

- via a custom nearest-neighbor CL

<br>

First framework to learn **domain-invariant** representation for UDA of TS data. 

<br>

# 1. Introduction

Need for effective domain adaptation of TS, to learn Domain-invariant representations 

<br>

### Unsupervised domain adaptation (UDA) 

Few works have focused on UDA of TS

Previous works 

- utilize a tailored feature extractor to capture temporal dynamics of MTS via RNNs, LSTM, CNNs …

- minimize the domain discrepancy of learned features via ..
  - adversarial-based methods (Purushotham et al., 2017; Wilson et al., 2020; 2021; Jin et al., 2022)
  - restrictions through metric-based methods (Cai et al., 2021; Liu \& Xue, 2021).

<br>

Transfer Learning

- pre-train a NN via CL to capture the contextual representation of TS from unlabeled source domain. 
  - BUT … operate on a labeled target domain, which is different from UDA. 

<br>

No method for UDA of TS

$$\rightarrow$$ propose a novel framework for UDA of TS based on CL ( = CLUDA )

<br>

Components: 

- (1) Adversarial training
  - to minimize the domain discrepancy between source & target domains
- (2) Semantic-preserving augmentations
- (3) Custom nearest-neighborhood CL
  - further align the contextual representation across source and target domains

<br>

Datasets 1

- WISDM (Kwapisz et al., 2011)
- HAR (Anguita et al., 2013)
- HHAR (Stisen et al., 2015)

$$\rightarrow$$ CLUDA leads to increasing accuracy on target domains by an important margin. 

<br>

Datasets 2

( two largescale real-world medical datasets )

- MIMIC-IV (Johnson et al., 2020)
- AmsterdamUMCdb (Thoral et al., 2021)

<br>

### Contributions

1. Propose CLUDA

   ( unsupervised domain adaptation of time series )

2. Capture domain-invariant, contextual representations in CLUDA 
   - via a custom approach combining nearest-neighborhood CL & adversarial learning

<br>

# 2. Related Work

## (1) Unsupervised Domain Adaptation (UDA)

Leverage **LABELED source** domain to predict **UNLABELED target** domain

Typically aim to minimize **domain discrepancy**

3 Categories

- (1) Adversarial-based

  - reduce domain discrepancy via **domain discriminator networks**

    ( force to learn domain-invariant feature representations )

- (2) Contrastive
  - via minimization of CL loss, aims to bring source & target embeddings of the same class
  - labels are UNKNONWN … rely on pseudo-labels
- (3) Metric-based

<br>

## (2) UDA for TS

### Variational recurrent adversarial deep domain adaptation (VRADA)

first UDA method for MTS that uses adversarial learning for reducing domain discrepancy.

-  Feature extractor = variational RNN
- trains the classifier and the domain discriminator (adversarially) for the last latent variable of its variational recurrent neural network. 

<br>

### Convolutional deep domain adaptation for time series (CoDATS) 

- same adversarial training as VRADA
- Feature extractor = CNN 

<br>

### Time series sparse associative structure alignment (TS-SASA) 

- metric-based method
- Intra-variables & inter-variables attention mechanisms are aligned between the domains via the minimization of maximum mean discrepancy (MMD). 

<br>

### Adversarial spectral kernel matching (AdvSKM) 

- metric-based method
- aligns the two domains via MMD. 
- ntroduces a spectral kernel mapping, from which the output is used to minimize MMD between the domains. 

<br>

$$\rightarrow$$ [ Common ] Aim to align the features across source and target domains.

<br>

***Research Gap***

Existing works merely align the features across source & target domains. 

Even though the source and target distributions overlap … this results in mixing the source and target samples of different classes. 

<br>

# 3. Problem Definition

Classification task 

2 distributions over the TS 

- from the source domain $$\mathcal{D}_S$$ 
- from the target domain $$\mathcal{D}_t$$

<br>

**Labeled** samples from the source domain given by $$\mathcal{S}=\left\{\left(x_i^s, y_i^s\right)\right\}_{i=1}^{N_s} \sim \mathcal{D}_S$$, 

**Unlabeled** samples from the target domain given by $$\mathcal{T}=\left\{x_i^t\right\}_{i=1}^{N_t} \sim \mathcal{D}_T$$

<br>

Multivariate TS : each $$x_i$$ is a sample of MTS, 

- denoted by $$x_i=\left\{x_{i t}\right\}_{t=1}^T \in \mathbb{R}^{M \times T}$$

<br>

Goal : build a classifier 

- that generalizes well over $$\mathcal{T}$$ 
- by leveraging the labeled $$\mathcal{S}$$. 

( At evaluation ) use the labeled $$\mathcal{T}_{\text {test }}=\left\{\left(x_i^t, y_i^t\right)\right\}_{i=1}^{N_{\text {test }}} \sim \mathcal{D}_T$$ 

<br>

# 4. Proposed CLUDA Framework

Overview of our CLUDA framework

1. Domain adversarial training, 
2. Capture the contextual representation
3. Align contextual representation across domains.

<br>

## (1) Architecture

![figure2](/assets/img/ts/img442.png)

**(1) Feature extractor** $$F(\cdot)$$

- takes the time series $$x^s$$ and $$x^t$$  & creates embeddings $$z^s$$ and $$z^t$$
- momentum updated feature extractor network $$\tilde{F}(\cdot)$$

**(2) Classifier network** $$C(\cdot)$$

- predict $$y^s$$ of TS from the source domain using the embeddings $$z^s$$.

**(3) Discriminator network** $$D(\cdot)$$ 

- trained to distinguish source $$z^s$$ from target $$z^t$$. 
- introduce domain labels
  - $$d=0$$ for source instances
  - $$d=1$$ for target instances

<br>

## (2) Adversarial Training for UDA

Minimize a combination of two losses: 

- (1) Prediction loss $$L_c$$ 

  - $$L_c=\frac{1}{N_s} \sum_i^{N_s} L_{\mathrm{pred}}\left(C\left(F\left(x_i^s\right)\right), y_i^s\right)$$.

- (2) Domain classification loss $$L_{\mathrm{disc}}$$ 

  - learn domain-invariant feature representations

  - use adversarial learning

  - $$D(\cdot)$$ is trained to minimize the domain classification loss

    & $$F(\cdot)$$ is trained to maximize the same loss

  - achieved by the gradient reversal layer $$R(\cdot)$$ between $$F(\cdot)$$ and $$D(\cdot)$$, 

    defined by $$R(x)=x, \quad \frac{\mathrm{d} R}{\mathrm{~d} x}=-\mathbf{I} $$

  - $$L_{\text {disc }}=\frac{1}{N_s} \sum_i^{N_s} L_{\text {pred }}\left(D\left(R\left(F\left(x_i^s\right)\right)\right), d_i^s\right)+\frac{1}{N_t} \sum_i^{N_t} L_{\text {pred }}\left(D\left(R\left(F\left(x_i^t\right)\right)\right), d_i^t\right) $$.

<br>

## (3) Capturing Contextual Representations

(1) Encourage $$F(\cdot)$$ to learn **label-preserving information** captured by the context. 

(2) Hypothesize that (a) < (b)

- (a) discrepancy between the contextual representations of two domains
- (b) discrepancy between their feature space

<br>

Leverage CL in form of MoCo

- apply semantic-preserving augmentations to each sample of MTS

- 2 views of each sample
  - query $$x_q$$ …… $$z_q=F\left(x_q\right)$$
  - key $$x_k$$ ……. $$z_k=\tilde{F}\left(x_k\right)$$

<br>

Momentum-updated feature extractor

- $$\theta_{\tilde{F}} \leftarrow m \theta_{\tilde{F}}+(1-m) \theta_F$$.

<br>

Contrastive loss

- $$L_{\mathrm{CL}}=-\frac{1}{N} \sum_{i=1}^N \log \frac{\exp \left(Q\left(z_{q i}\right) \cdot z_{k i} / \tau\right)}{\exp \left(Q\left(z_{q i}\right) \cdot z_{k i} / \tau\right)+\sum_{j=1}^J \exp \left(Q\left(z_{q i}\right) \cdot z_{k j} / \tau\right)}$$.

<br>

Since we have two domains (i.e., source and target)

$$\rightarrow$$ two CL loss ( $$L_{\mathrm{CL}}^s$$ & $$L_{\mathrm{CL}}^t$$  )

<br>

## (4) Aligning the Contextual Representation Across Domains

Further aligns the contextual representation across the source and target domains

First nearest-neighbor CL approach for UDA of TS



Nearest-neighbor contrastive learning (NNCL) 

- facilitate the classifier $$C(\cdot)$$ to make accurate predictions for the target domai
- by creating positive pairs between domains
  - explicitly align the representations across domains. 

<br>

$$L_{\mathrm{NNCL}}=-\frac{1}{N_t} \sum_{i=1}^{N_t} \log \frac{\exp \left(z_{q i}^t \cdot N N_s\left(z_{k i}^t\right) / \tau\right)}{\sum_{j=1}^{N_s} \exp \left(z_{q i}^t \cdot z_{q j}^s / \tau\right)}$$.

- retrieves the nearest-neighbor of an embedding from the source queries $$\left\{z_{q i}^s\right\}_{i=1}^{N_s}$$.

<br>

## (5) Training

Final Loss :

$$L=L_c+\lambda_{\mathrm{disc}} \cdot L_{\mathrm{disc}}+\lambda_{\mathrm{CL}} \cdot\left(L_{\mathrm{CL}}^s+L_{\mathrm{CL}}^t\right)+\lambda_{\mathrm{NNCL}} \cdot L_{\mathrm{NNCL}}$$.

<br>

# 5. Experimental Setup

Earlier works of UDA on time series 

- Wilson et al., 2020; 2021; Cai et al., 2021; Liu \& Xue, 2021

<br>

## (1) Datasets

1. Established benchmark datasets

   - WISDM (Kwapisz et al., 2011), HAR (Anguita et al., 2013), and HHAR (Stisen et al., 2015). 
   - each patient = each doomain
   - randomly sample 10 source-target domain pairs for evaluation. 

2. Real-world setting with medical datasets

   - MIMIC-IV (Johnson et al., 2020) and AmsterdamUMCdb (Thoral et al., 2021). 

   - treat each age group as a separate domain

<br>

## (2) Baselines

Model w/o UDA

- use feature extractor $$F(\cdot)$$ and the classifier $$C(\cdot)$$ using the same architecture as in our CLUDA. 
- only trained on the source domain.

<br>

Model w/ UDA ( for TS )

- (1) VRADA (Purushotham et al., 2017)
- (2) CoDATS (Wilson et al., 2020)
- (3) TS-SASA (Cai et al., 2021)
- (4) AdvSKM (Liu \& Xue, 2021)

<br>

Model w/ UDA ( not CV )

- (5) CAN (Kang et al., 2019)
- (6) CDAN (Long et al., 2018)
- (7) DDC (Tzeng et al., 2014)
- (8) DeepCORAL (Sun \& Saenko, 2016)
- (9) DSAN (Zhu et al., 2020)
- (10) HoMM (Chen et al., 2020a)
- (11) MMDA (Rahman et al., 2020). 

<br>

#  6. Results

## (1) Established benchmark datasets

![figure2](/assets/img/ts/img443.png)

Average accuracy of each method for 10 source-target domain pairs

- on the WISDM, HAR, and HHAR datasets

<br>

![figure2](/assets/img/ts/img444.png)

- to study the domain discrepancy 

- (a) The embeddings of w/o UDA

  - significant domain shift between source and target
  - two clusters of each class (i. e., one for each domain)

- (b) CDAN as the best baseline

  - reduces the domain shift 

  - by aligning the features of source and target for some classes, 

    ( BUT mixes the different classes of the different domains (e.g., blue class of source and green class of target overlap). )

- (c) CLUDA
  - pulls together the source (target) classes for the source (target) domain (due to the CL)
  - pulls both source and target domains together for each class (due to the alignment).

<br>

### Ablation Study

Figure 2-(b)

<br>

## (2) Real-world setting with medical datasets


![figure2](/assets/img/ts/img445.png)
