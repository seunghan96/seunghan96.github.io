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

  $$\rightarrow$$  Can be challenging!!

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

  $$\rightarrow$$ Robustness of the model is improved!

<br>

### c) GAN

**CausalGAN** = 2-stage causal GAN

- Kocaoglu et al. [166] 

- Stage 1) Trains a causal implicit GAN on binary labels
- Stage 2) New conditional GAN to help the generator sample from the correct intervention distribution. 

![figure2](/assets/img/tab/img78.png)

<br>

**Scalable generative causal adversarial network (CAN)**

- Moraffah et al. [167]

- Limitation of Causal GAN

  = Causal graph constructed in CausalGAN **relies on known labels!**

- CAN: Learns the causal relations ***from the data iteslf!***

- Structured into 2 parts: 
  - (1) **Label generating network (LGN)**
    - Learns causal relationships from data and generates samples
  - (2) **Conditional image generating network (CIGN)**
    - Receives labels and generates the corresponding images

![figure2](/assets/img/tab/img79.png)

<br>

**Causal generative neural networks (CGNNs)** 

- Goudet et al. [168]

- Learn data distributions with causal construction generators

<br>

**Causal-TGAN**

- Wen et al. [169] 

- Goal: Generate **synthetic tabular data** using the tabular data’s causal information

- Multiple causal processes are captured by building an **SCM**

<br>

**SCIGAN**: Hierarchical discriminator 

- Bica et al. [170] 

- Estimatie counterfactual outcomes at successive interventions. 

- Goal: Estimatie counterfactual outcomes at successive interventions. 

- How? Significantly modified GAN model 

  - Generate counterfactual outcomes

    $$\rightarrow$$ Used to learn an inference model ( with standard supervised methods )

![figure2](/assets/img/tab/img80.png)

<br>

## (2) Contrastive Learning (CL)

### a) Supervised contrastive learning

$$C^2L$$: Causal-based CL 

- Choi et al. [171] 

- To improve the robustness of **"text categorization"** models
- Candidate tokens are selected based on **attribution scores**
- Causality of these candidate tokens 
  - Verified by evaluating their individualized treatment effect (ITE)

![figure2](/assets/img/tab/img81.png)

<br>

**Proactive Pseudo-Intervention (PPI)**

- Wang et al. [172] 

- Causal intervention-based CL (for **visual problems**)

- Pseudo-interventions are synthesized from observational data using CL

  $$\rightarrow$$ Reduces the model’s dependence on image features that are strongly correlated with the target label but not causally related

- Result: Addresses the issue of DNNs over-relying on non-causal visual informationin image classification 

![figure2](/assets/img/tab/img82.png)

<br>

### b) Self-supervised contrastive learning

**Graph contrastive invariant learning (GCIL)**

- Mo et al. [173] 

- Graph generation based on the SCM

- Limitation of previous works: Traditional graph CL is affected by non-causal information

- GCIL

  - Uses an SCM to describe the graph generation process

  - Original graph $$G$$: Divided into ..

    - (1) A set of causal variables $$C$$ 
    - (2) A set of non-causal variables $$S$$

  - Intervene causally on the noncausal variable $$S$$ 

    $$\rightarrow$$ T ensure that the variable satisfies the following equation:

    $$P^{d o\left(S=s_i\right)}(Y \mid C)=P^{d o\left(S=s_j\right)}(Y \mid C)$$.

- Summary: Generates causal views to model interventions on non-causal factors from a graph perspective

<br>

## (3) Diffusion Models

## (4) Reinforcement Learning

## (5) Recommendation Algorithm





