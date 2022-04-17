---
title: Introduction to GNN - Chapter 6) Graph Attention Networks (GATs)
categories: [GNN]
tags: []
excerpt: 

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# 6. Graph Attention Networks (GATs)

GCN : treates all neighbors **EQUALLY**

GAT : assing **DIFFERENT ATTENTION SCORE**

<br>

2 varinants

- (1) GAT
- (2) GAAN

<br>

## 6-1. GAT

attention mechanism of node pair $$(i,j)$$

- $$\alpha_{i j}=\frac{\exp \left(\operatorname{LeakyReLU}\left(\mathbf{a}^{T}\left[\mathbf{W h}_{i}  \mid \mid  \mathbf{W h}_{j}\right]\right)\right)}{\sum_{k \in N_{i}} \exp \left(\operatorname{LeakyReLU}\left(\mathbf{a}^{T}\left[\mathbf{W h}_{i}  \mid \mid  \mathbf{W h}_{k}\right]\right)\right)}$$.

<br>

final output features of each node :

- $$\mathbf{h}_{i}^{\prime}=\sigma\left(\sum_{j \in N_{i}} \alpha_{i j} \mathbf{W h}_{j}\right) $$.

<br>

Multi-head Attention

- apply $$K$$ independent attention mechanism

<br>

Concatenate ( or Average features )

- ( concatenate )
  - $$\mathbf{h}_{i}^{\prime} = \mid \mid _{k=1}^{K} \sigma\left(\sum_{j \in N_{i}} \alpha_{i j}^{k} \mathbf{W}^{k} \mathbf{h}_{j}\right)$$.
- ( average )
  - $$\mathbf{h}_{i}^{\prime} =\sigma\left(\frac{1}{K} \sum_{k=1}^{K} \sum_{j \in N_{i}} \alpha_{i j}^{k} \mathbf{W}^{k} \mathbf{h}_{j}\right)$$.

<br>

![figure2](/assets/img/gnn/img343.png)

<br>

Properties of GAT

- (1) parallizeable ( efficient )

- (2) can deal with nodes with different degrees

  & assign correspoding weights to their neighbors

- (3) can be applied to inductive learning problems

$$\rightarrow$$ outperforms GCN!

<br>

## 6-2. GAAN

also uses **multi-head attention**

<br>

GAT vs GaAN : for computing attention coefficients…

- (1) GAT : use FC layer
- (2) GaAN : uses **key-value attention** & **dot product attention**

<br>Assigns different weights for different heads,

by computing **additional soft gate** ( = gated attention aggregator )

<br>