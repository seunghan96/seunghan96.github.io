---
title: Introduction to GNN - Chapter 1) Introduction
categories: [GNN]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# 1. Introduction

Graphs : 

- data structure with (1) node & (2) edges

- in **non-Euclidean** space

- tasks

  - (1) node classification

  - (2) link prediction
  - (3) clustering

- convincing performance & high interpretability



## 1.1 Motivation

### 1.1.1 CNN

GNNs are motivated by CNNs

Keys of CNN

- (1) local connection
- (2) shared weights
- (3) use of multi-layer

<br>

Graphs & Keys of CNN

- (1) graphs are the most typical **locally connected structure**
- (2) shared weights **reduce computational cost**, compared with traditional spectral graph theory
- (3) multi-layer structre is the key to deal with **hierarchical patterns**

<br>

CNN vs GNN

- CNN : can only operate on **regular Euclidean space**
- GNN : ~ **non-Euclidean**

![figure2](/assets/img/gnn/img336.png)

<br>

### 1.1.2 Network Embedding

motivation also comes from **graph embedding**

- traiditional ML : rely on **hand-engineered features**

  ( thus, limited by its **inflexibility** & **high cost** )

- Example)

  - DeepWalk node2vec, LINE, TADW,...

  $$\rightarrow$$ Drawbacks :

  ​	(1)  **no parameters are shared....INEFFICIENCY**

  ​	(2) **lack of generalization ability** ( can not deal with dynamic graphs & new graphs )

<br>

## 1-2. Overview

provide an introduction to **different GNN models**

- (1) detailed review over exisiting **GNN models**
- (2) categorize the **applications**