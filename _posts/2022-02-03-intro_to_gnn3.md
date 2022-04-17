---
title: Introduction to GNN - Chapter 3) Vanilla GNN
categories: [GNN]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# 3. Vanilla GNN

- introduction of GNN

- limitations of GNN

  ( in representation capability & training efficiency )

<br>

## 3-1. Introduction

target of GNN

- learn **state embedding** $$\mathbf{h}_{v} \in \mathbb{R}^{s}$$

  ( encodes information of neighbors )

- state embeding is used to produce an **output $$\mathbf{o}_{v}$$**

  ( ex. predicted node label )

<br>

Vanilla GNN

- deals with **UNdirected**, **HOMOgeneous** graphs
- features
  - node features $$\mathbf{x}_{v}$$
  - edge features ( optional )

<br>

Notation 

- set of EDGES : $$c o[v]$$ 
- set of NEIGHBORS : $$n e[v]$$ 

<br>

## 3-2. Model

given **input features** ( of nodes & edges )...

update the **node state**, according to **input neighborhood**

<br>

### Vector

Notation

- $$f$$ : **local transition** function
- $$g$$ : **local output** function

<br>

Model

- $$\mathbf{h}_{v}=f\left(\mathbf{x}_{v}, \mathbf{x}_{c o[v]}, \mathbf{h}_{n e[v]}, \mathbf{x}_{n e[v]}\right)$$.
- $$\mathbf{o}_{v}=g\left(\mathbf{h}_{v}, \mathbf{x}_{v}\right)$$.

<br>

### Matrix 

Notation

- $$\mathbf{H}$$ : states
- $$\mathbf{O}$$ : output
- $$\mathbf{X}$$ : features ( edge & nodes )
- $$\mathbf{X}_{N}$$ : features ( nodes )

<br>

Model

- $$\mathbf{H} =F(\mathbf{H}, \mathbf{X})$$.
- $$\mathbf{O} =G\left(\mathbf{H}, \mathbf{X}_{N}\right)$$.

<br>

### iterative Scheme

$$\mathbf{H}^{t+1}=F\left(\mathbf{H}^{t}, \mathbf{X}\right)$$.

- converges exponentially fast, for any initival value $$\mathbf{H}(0)$$

<br>

### Learning

with the target information ( label : $$t_v$$ )

- $$\text { loss }=\sum_{i=1}^{p}\left(\mathbf{t}_{i}-\mathbf{o}_{i}\right)$$.

<br>

## 3-3. Limitations

limitations of GNN

1. computationally inefficient to update the hidden states of nodes iteratively

   $$\rightarrow$$ needs $$T$$ steps of computation, to approximate the fixed point

2. same parameters in the iteration

3. some informative features on edges may not be efficiently used

   ( ex. edges in KG (Knowledge Graph) )

4. if $$T$$ Is large.... unsuitable to use fixed points

<br>

Several variants are proposed!

- ex) GGNN (Gated GNN) : to solve limitation (1)
- Ex) R-GCN (Relational GCN) : to deal with **directed** graph

<br>