---
title: Linear Attention
categories: [ML]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<br>

# 1. Limitation of Self-Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V$$.

- $$Q \in \mathbb{R}^{n \times d}$$: Query
- $$K \in \mathbb{R}^{n \times d}$$: Key
- $$V \in \mathbb{R}^{n \times d_v}$$: Value
- $$n$$: Length of sequence

<br>

Complexity: $$\mathcal{O}(n^2)$$ → Due to $$QK^T$$ 

<br>

# 2. Linear Attention

Complexity:  $$\mathcal{O}(n^2)$$ $$\rightarrow$$ $$\mathcal{O}(n)$$

<br>

핵심 아이디어

- **softmax($$QK^T$$ )**를 직접 계산하지 말고,
- $$Q$$와 $$K$$에 **kernel 함수를 적용한 변형된 형태**로 근사!

<br>

Softmax Attention: $$A = \text{softmax}(QK^T)V$$

Linear Attention: $$A = \phi(Q) \left( \phi(K)^T V \right)$$.

- $$\phi(\cdot)$$: Non-linear function
- $$\phi(Q) \in \mathbb{R}^{n \times d}$$.
- $$\phi(K)^T V \in \mathbb{R}^{d \times d_v}$$: 한 번만 계산

<br>

|                   | Complexity             | Memory               |
| ----------------- | ---------------------- | -------------------- |
| Softmax Attention | $$\mathcal{O}(n^2 d)$$ | $$\mathcal{O}(n^2)$$ |
| Linear Attention  | $$\mathcal{O}(nd^2)$$  | $$\mathcal{O}(nd)$$  |

![figure2](/assets/img/ts/img793.png)

