---
title: Causal Inference Meets Deep Learning; A Comprehensive Survey - Part 3
categories: [ML, CI]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Causal Inference Meets Deep Learning: A Comprehensive Survey

<br>

# 5. DL with Causal Inference

### Causal discovery 

= ***Inferring causal relationships*** from ***data***

<br>

### DL + CI

Traditional DL

- Relies on large IID datasets
- Focus on correlation rather than causation
  - Correlation-based models: Unstable and sensitive to small data changes

<br>

<br>

Causal learning 

- Improve generalization, stability, and interpretability in DL

<br>

Causal Inference + DL

![figure2](/assets/img/tab/img77.png)

<br>

## (1) Adversarial Learning 

- Goal: Maintaining **model stability/robustness**
  - In the presence of **malicious environments**
- How? By ***adding perturbations*** to real samples
  - NN will misled in their judgments by them!
- Divided into 
  - (1) Adversarial **attacks**
  - (2) Adversarial **defenses**

<br>

### a) Adversarial attack

Aim to **generate adversarial samples** more efficiently

- By manipulating the input samples

<br>

Adversarial attack + Causal Inference

- Testing for conditional independence in a **multi-dimensional** dataset

  $\rightarrow$  Can be challenging!!

- Adversarial attack the IID distn

  - **[Humans]** May not perceive these changes!
  - **[Model]** Modified samples can cause intervention for the model!

- ***Causality*** can mitigate the **impact of adversarial attacks on DL**

<br>

Ren et al. [161]

- Employ **Transformer** as a tool to construct a **causal model**
  - Model = Explains the generation and performance of adversarial samples
- Propose a simple and effective strategy to defend against adversarial attacks

<br>

Cai et al. [162] 

- Novel adversarial learning framework based on the **causal generation method**
- Generates counterfactual adversarial examples ...
  - By altering the distn through **intervening variables** 

<br>

### b) Adversarial defense

Aims to offer more effective protection against adversarial samples

<br>

Zhang et al. [163] 

- Cause of adversarial vulnerability in DNNs = **Model’s reliance on "false correlations"**

- Construct **causal graphs** 

  - To model the generation process of adversarial samples

- Propose an **adversarial distribution alignment method** 

  - To formalize the intuition behind adversarial attacks. 

- Eliminate the differences between natural & adversarial distns

  $\rightarrow$ Robustness of the model is improved!

<br>

### c) GAN

Kocaoglu et al. [166] 

- CausalGAN = 2-stage causal GAN
  - Stage 1) Trains a causal implicit generative model on binary labels
  - Stage 2) New conditional GAN to help the generator sample from the correct intervention distribution. 

<br>

- Moraffah et al. [167]argued that the causal graph constructed in CausalGAN relieson known labels, making it challenging to apply the model toreal-world tasks and difficult to scale with large amounts oflabeled data. To address these issues, they proposed a scalablegenerative causal adversarial network (CAN). CAN is structuredinto 2 parts: the label generating network (LGN), which learnscausal relationships from data and generates samples, and theconditional image generating network (CIGN), which receiveslabels and generates the corresponding images. Goudet et al.Model Publication Key characteristics Deep algorithm Causal methodDOVI [183] NeurIPS 2021 Confounded observationaldata, causal reinforcementlearningReinforcement learning Causal intervention- [184] ACC 2023 Gene regulatory networks,reinforcement learning,causal inferenceReinforcement learning Causal interventionDeep- Deconf [189] ACM 2022 Recommender systems,causal inference, multi-causeconfoundersRecommendation algorithm Causal inferenceCountER [186] ACM 2021 Explainable recommendation,counterfactual explanation,counterfactual reasoning,machine learning, explainableAIRecommendation algorithm CounterfactualCEF [187] ACM 2022 Explainable fairness,recommender systems,explainable recommendation,fairness in AI, counterfactualreasoningRecommendation algorithm CounterfactualCauser [188] 2022 Sequential recommendation,causal behavior discoveryRecommendation algorithm Causal discoveryTable 1. (Continued)

[168] introduced a framework called causal generative neuralnetworks (CGNNs) to learn data distributions with causalconstruction generators. Wen et al. [169] proposed a datageneration architecture called Causal-TGAN, which aims tosolve the causal problem in tabular data generation and gener-ate datasets with different variable types. Multiple causal pro-cesses are captured by building an SCM, which improves theaccuracy of the target data distribution. Bica et al. [170] pro-posed a hierarchical discriminator called SCIGAN, for estimat-ing counterfactual outcomes at successive interventions. Thekey idea is to generate counterfactual outcomes through amodified GAN model and learn an inferential model using astandard supervised approach to estimate counterfactuals fornew samples.



## (2) Contrastive Learning

## (3) Diffusion Models

## (4) Reinforcement Learning

## (5) Recommendation Algorithm





