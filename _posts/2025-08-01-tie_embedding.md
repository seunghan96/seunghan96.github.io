---
title: Tie Embeddings
categories: [LLM, MULT]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Tie Embeddings

<br>

# 1. 개념

***“embedding layer와 LM head를 tie한다(tied weights)”***

- **(Input) embedding matrix** $$E \in \mathbb{R}^{V \times d}$$
- **(Output) LM head** $$W \in \mathbb{R}^{d \times V}$$

$$\rightarrow$$ 이 둘의 parameter sharing!

$$\rightarrow$$ 보통 $$W = E^\top$$ 로 묶음 (=tie)

<br>

Notation

- $$V$$: vocabulary size

- $$d$$: hidden/embedding dim

- Input token $$x_t$$ → embedding: $$e(x_t) = E[x_t] \in \mathbb{R}^{d}$$.

- Decoder의 마지막 hidden vector $$h_t \in \mathbb{R}^{d}$$에서 **logit**:

  - **untied**: $$z_t = h_t W + b$$.
  - **tied**: $$z_t = h_t E^\top + b$$.

  ( 여기서 $$b \in \mathbb{R}^{V}$$ (bias)는 보통 **별도로 둔다** )

<br>

장점

- **파라미터 수 감소**
- **학습 안정/일관성 향상**
- **성능 향상**

<br>

# 2. 코드

```python
import torch, torch.nn as nn

V, d = 50_000, 1024

emb = nn.Embedding(V, d)
lm_head = nn.Linear(d, V, bias=True)

# tie: lm_head.weight를 emb.weight와 공유(전치 X, PyTorch Linear는 (out,in))
lm_head.weight = emb.weight  # 이 경우 shape가 (V, d) vs (V, d)로 맞아야 함
# 통상적으로는 logit을 h @ E^T로 계산하므로, Linear 없이 matmul을 직접 쓰기도!
```

실무에서는 보통 ..

- (1) **직접 전치 곱**으로 logits을 계산하거나

- (2) `lm_head.weight = emb.weight`로 두되 **모델 forward에서 전치해 사용**하는 식으로

<br>

### HuggingFace 스타일

```python
class TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(...)  # Transformer 등
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
				
        #----------------------------------------------#
        # tie
        self.lm_head.weight = self.embed.weight
        #----------------------------------------------#
        
    def forward(self, input_ids):
        x = self.embed(input_ids)   # (B, T, d)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)    # (B, T, V) 
        return logits
```

- HF 트랜스포머에서는 `model.tie_weights()`가 내부에서 이런 바인딩을 처리합니다.

- 토크나이저 리사이즈 시(`model.resize_token_embeddings`)도 **tie를 유지**하도록 자동 처리됩니다.

<br>

# References

- **Inan et al., 2016** “Tying Word Vectors and Word Classifiers”
- **Press & Wolf, 2017** “Using the Output Embedding to Improve Language Models”