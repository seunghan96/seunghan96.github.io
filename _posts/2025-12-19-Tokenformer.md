---
title: TokenFormer; Rethinking Transformer Scaling with Tokenized Model Parameters
categories: [LLM, NLP, TS]
tags: []
excerpt: ICLR 2025
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## TokenFormer: Rethinking Transformer Scaling with Tokenized Model Parameters

```
Wang, Haiyang, et al. "TokenFormer: Rethinking Transformer Scaling with Tokenized Model Parameters." ICLR 2025
```

참고: 

- https://aipapersacademy.com/tokenformer/
- https://arxiv.org/abs/2410.23168

<br>

### Contents

1. Motivation
2. Transformer vs. Tokenformer
   1. Transformer
   2. Tokenformer

3. Experiments

<br>

# 1. Motivation

Training Transformers ***from scratch*** becomes increasingly costly!

![figure2](/assets/img/llm/img189.png)

<br>

# 2. Transformer vs. Tokenformer

![figure2](/assets/img/llm/img190.png)

<br>

## (1) Transformer 

### Procedure

- Step 1) Linear projection

  = Input first passes through a linear projection block

  $$\rightarrow$$ Generate inputs for the attention block ($$Q,K,V$$)

  $$\rightarrow$$ Interactions between the parameters and the input tokens

- Step 2) Self-attention 

  = Allows input tokens to interact with each other

- Step 3) FFN

  = Interactions between tokens and parameters

<br>

## (2) Tokenformer

### Key Idea

**Token-parameter** interactions 

- Calculated via linear projection (fixed size of parameters)

  $$\rightarrow$$ Necessitating training from scratch when increasing model size :(

<br>

Solution? 

Create a **fully attention-based model**, including **token-parameter** interactions

$$\rightarrow$$ More flexible architecture that supports **incremental \# parameter increases**



### Procedure

- Step 1) Feed input tokens to **token-parameter attention** block
  - [Input] Input tokens (= Query) & ***parameters*** (= Key param, Value param)
  - [Output] Used as the inputs for the self-attention block (Q, K, and V)
- ( Step 2 = same as Transformer )
- Step 3) Replace FFN with **token-parameter attention** block
  - Query: Output from the self-attention block
  - K,V: Different parameters

<br>

$$\text { Attention }(Q, K, V)=\operatorname{softmax}\left[\frac{Q \cdot K^{\top}}{\sqrt{d}}\right] \cdot V$$.

$$\text { Pattention }\left(X, K_P, V_P\right)=\Theta\left(X \cdot K_P^{\top}\right) \cdot V_P$$.

<br>

# 3. Experiments

![figure2](/assets/img/llm/img192.png)
