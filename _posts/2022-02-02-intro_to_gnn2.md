---
title: Introduction to GNN - Chapter 2) Basic of Math & Graph
categories: [GNN]
tags: []
excerpt: 

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# 2. Basic of Math & Graph

## 2-1. Linear Algebra

### 2-1-1. Eigendecomposition

Notation : 

- $$\mathbf{A}$$ : matrix in $$\mathbb{R}^{n \times n}$$ 

- $$\mathbf{v} \in \mathbb{C}^{n}$$  ( non-zero vector ) is **EIGENVECTOR** of $$\mathbf{A}$$ ,

  if there exists such scalar $$\lambda \in \mathbb{C}$$ that $$\mathbf{A v}=\lambda \mathbf{v}$$

  ( $$\lambda$$ : **EIGENVALUE** )

<br>

Matrix Notation

$$\mathbf{A}\left[\begin{array}{llll}
\mathbf{v}_{1} & \mathbf{v}_{2} & \ldots & \mathbf{v}_{n}
\end{array}\right]=\left[\begin{array}{llll}
\mathbf{v}_{1} & \mathbf{v}_{2} & \ldots & \mathbf{v}_{n}
\end{array}\right]\left[\begin{array}{cccc}
\lambda_{1} & & & \\
& \lambda_{2} & & \\
& & \ddots & \\
& & \lambda_{n}
\end{array}\right] $$.

<br>

**Eigenvalue Decomposition**

If $$\mathbf{V}=\left[\begin{array}{llll}\mathbf{v}_{1} & \mathbf{v}_{2} & \ldots & \mathbf{v}_{n}\end{array}\right]$$ are **LINEARLY INDEPENDENT**....

( = **INVERTIBLE** matrix )

$$\mathbf{A}=\mathbf{V} \operatorname{diag}(\lambda) \mathbf{V}^{-1}$$.

- $$\mathbf{V}=\left[\begin{array}{llll}\mathbf{v}_{1} & \mathbf{v}_{2} & \ldots & \mathbf{v}_{n}\end{array}\right]$$,

$$\mathbf{A}=\sum_{i=1}^{n} \lambda_{i} \mathbf{v}_{i} \mathbf{v}_{i}^{T}$$.

<br>

***Caution***

- not all square matrix can be decomposed like that!

  should be invertible! ( = should have $$n$$ lindear INDEPENDENT eigenvectors )

- SYMMETRIC matrix has an eigendecomposition!

<br>

### 2-1-2. Singular Value Decomposition (SVD)

Notation

- $$r$$ : rank of $$A^TA$$

  = means that ***there exists $$r$$ positive scalars*** $$\sigma_{1} \geq \sigma_{2} \geq \cdots \geq \sigma_{r}>0$$,

  such that for $$1 \leq i \leq r, \mathbf{v}_{i}$$ is an eigenvector of $$\mathbf{A}^{T} \mathbf{A}$$ with corresponding eigenvalue $$\sigma_{i}^{2}$$

  ( they are all LINEARLY INDEPENDENT! )

<br>

SVD : $$\mathbf{A}=U \Sigma V^{T}$$

- $$U \in \mathbb{R}^{m \times m}$$.
- $$V \in \mathbb{R}^{n \times n}$$.
- $$\Sigma \in \mathbb{R}^{m \times n}$$.
  - $$\Sigma_{i j}= \begin{cases}\sigma_{i} & \text { if } i=j \leq r, \\ 0 & \text { otherwise }\end{cases}$$.

<br>

SVD details

- column vectors of $$U$$ : eigenvectors of $$AA^T$$
- column vectors of $$V$$ : eigenvectors of $$A^TA$$

<br>

## 2-2. Graph Theory

### 2-3-1. Basic Concepts

Notation

- graph : $$G = (V,E)$$.
- degree of vertice $$v$$ : $$d(v)$$

<br>

Types of Graphs : directed & undirected

<br>

### 2-3-2. Algebra Representations of Graphs

[ 1. Adjacency Matrix : $$A \in \mathbb{R}^{n \times n}$$ ] 

$$A_{i j}= \begin{cases}1 & \text { if }\left\{v_{i}, v_{j}\right\} \in E \text { and } i \neq j \\ 0 & \text { otherwise. }\end{cases}$$.



<br>

[ 2. Degree Matrix : $$D \in \mathbb{R}^{n \times n}$$ ]

$$D_{i i}=d\left(v_{i}\right) $$.

<br>

[ 3. Laplacian Matrix : $$L \in \mathbb{R}^{n \times n}$$ ]

- UNdirected graph

$$L=D-A $$.

$$L_{i j}= \begin{cases}d\left(v_{i}\right) & \text { if } i=j \\ -1 & \text { if }\left\{v_{i}, v_{j}\right\} \in E \text { and } i \neq j \\ 0 & \text { otherwise. }\end{cases}$$.

<br>

[ 4. Symmetric Normalized Laplacian ]

$$\begin{aligned}
L^{s y m} &=D^{-\frac{1}{2}} L D^{-\frac{1}{2}} \\
&=I-D^{-\frac{1}{2}} A D^{-\frac{1}{2}} .
\end{aligned}$$.

where $$L_{i j}^{s y m}= \begin{cases}1 & \text { if } i=j \text { and } d\left(v_{i}\right) \neq 0 \\ -\frac{1}{\sqrt{d\left(v_{i}\right) d\left(v_{j}\right)}} & \text { if }\left\{v_{i}, v_{j}\right\} \in E \text { and } i \neq j \\ 0 & \text { otherwise. }\end{cases}$$.

<br>

[ 5. Random Walk Nomalized Laplacian ]

$$L^{r w}=D^{-1} L=I-D^{-1} A $$.

where $$L_{i j}^{r w}= \begin{cases}1 & \text { if } i=j \text { and } d\left(v_{i}\right) \neq 0 \\ -\frac{1}{d\left(v_{i}\right)} & \text { if }\left\{v_{i}, v_{j}\right\} \in E \text { and } i \neq j \\ 0 & \text { otherwise. }\end{cases}$$

<br>

[ 6. Incidence Matrix : $$M \in \mathbb{R}^{n \times m}$$ ]

- $$n$$ : number of nodes
- $$m$$ : number of edges

(1) directed graph

$$M_{i j}= \begin{cases}1 & \text { if } \exists k \text { s.t } e_{j}=\left\{v_{i}, v_{k}\right\} \\ -1 & \text { if } \exists k \text { s.t } e_{j}=\left\{v_{k}, v_{i}\right\} \\ 0 & \text { otherwise. }\end{cases}$$.

<br>

(2) undirected graph

$$M_{i j}= \begin{cases}1 & \text { if } \exists k \text { s.t } e_{j}=\left\{v_{i}, v_{k}\right\} \\ 0 & \text { otherwise. }\end{cases}$$.

<br>

