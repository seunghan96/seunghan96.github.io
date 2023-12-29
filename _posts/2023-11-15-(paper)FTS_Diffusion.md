---
title: Generative Learning for Financial TS with Irregular and Scale-Invariant Patterns
categories: [TS,GAN]
tags: []
excerpt: ICLR 2024 (?)
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Generative Learning for Financial TS with Irregular and Scale-Invariant Patterns

<br>

![figure2](/assets/img/ts/img541.png)

# Contents

0. Abstract
0. Introduction

<br>

# Abstract

Limited data in financial applications

$\rightarrow$ Synthesize **financial TS** !!

- Challenges: **Irregular & Scale-invariant** patterns

( Existing approaches: assume **regularity & uniformity** )

<br>

### FTS-Diffusion

To model **Irregular & Scale-invariant** patterns that consists of 3 modules

- **(1. Patterrn Recognition)** Scale-invariant pattern recogntion algorithm
  - to extract recurring patterns that vary in duration & magnitude
- **(2. Pattern Generation)** Diffusion-based generative network
  - to synthesize segments of patterns
- **(3. Pattern Evolution)** 
  - model the temporal transition of patterns
  - 
