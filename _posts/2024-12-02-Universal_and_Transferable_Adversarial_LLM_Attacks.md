---
title: Universal and Transferable Adversarial LLM Attacks
categories: [LLM, NLP]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Universal and Transferable Adversarial LLM Attacks

```
Zou, Andy, et al. "Universal and transferable adversarial attacks on aligned language models." arXiv preprint arXiv:2307.15043 (2023).
```

( https://arxiv.org/pdf/2307.15043 )

참고: 

- https://aipapersacademy.com/llm-attacks/
- https://tuananhbui89.github.io/blog/2024/paper-llm-attacks/

<br>

### Contents

1. Attacking LLMs
1. Overall Framework
3. How are suffixes created?
   1. Producing **Affirmitices Responses**
   2. Greedy Coordinate Gradient-based Search


<br>

# 1. Attacking LLMs

LLM는 **offensive 표현을 삼가**하도록 학습이 되어있음 (**aligned**)

$$\rightarrow$$ How to attack LLM?

<br>

### Jailbreaks

LLM로 하여금 **offensive 표현**하도록 유도하는 것!

- 방법 1) Human crafted prompts

  - 단점: require significant effort

- 방법 2) Automatic prompt-tuning for adversarial attacks

- 방법 3) 이 논문

  $$\rightarrow$$ A new class of attacks based on **automatically created suffixess**

<br>

```
Prompt: Tell me how to build a bomb. <enter generated suffix here>
```

<br>

# 2. Overall Framework

특징 요약

- (1) Universal: 다양한 prompt에서 작동한다
- (2) Transferable: 다양한 model에서 작동한다

<br>

![figure2](/assets/img/llm/img93.png)

<br>

# 3. How are suffixes created?

Opensource: https://github.com/llm-attacks/llm-attacks



## (1) Producing **Affirmitices Responses**

대답을 유도하기 위해, 모델로 하여금 "Sure, here is..."로 시작하도록 유도함!

$$\rightarrow$$ 이를 유도하는 loss function을 사용

<br>

Notation

- Token: $$x_i \in\{1, \ldots, V\}$$ (where $$V$$ denotes the vocabulary size = number of tokens) 
- Next token prediction: $$p\left(x_{n+1} \mid x_{1:n}\right)$$
  - for any $$x_{n+1} \in\{1, \ldots, V\}$$,

- Autoregressive LM: $$p\left(x_{n+1: n+H} \mid x_{1-n}\right)=\prod_{i=1}^H p\left(x_{n+i} \mid x_{1=n+i-1}\right)$$

<br>

### Adversarial loss

(Negative log) probability of some target sequences of tokens $$x_{n+1:n+H}^*$$ 

$$\mathcal{L}\left(x_{1: n}\right)=-\log p\left(x_{n+1: n+H}^* \mid x_{1: n}\right)$$.

- $$x_{1:n}$$ : (ADV prompt를 포함한) 전체 입력
- $$x_{n+1:n+H}^*$$: (예시) "Sure, here is how to build a bomb.")

<br>

Adversarial suffix에 대한 Optimization: 

- $$\operatorname{minimize}_{x_x \in\{1, \ldots V\}^{\mid \mathcal{I} \mid}} \mathcal{L}\left(x_{1:n}\right)$$.
  - where $$\mathcal{I} \subset\{1, \ldots, n\}$$ denotes the indices of the adversarial suffix tokens in the LLM input.
- 이를 최소화하는 ADV Prompt를 찾자!

<br>

## (2) Greedy Coordinate Gradient-based Search

![figure2](/assets/img/llm/img94.png)

Multiple token replacement steps (based on the above loss function)

- Step 1) (ADV Prompt token들 중에서) 위의 loss를 **"최대화"**하는 top-K개의 token을 뽑는다.
- Step 2) (총 B번 반복) 데이터 내의 일부 토큰을 위의 토큰과 replace.
- Step 3) (총 B개 중) 위의 loss를 **"최소화"**하는 데이터를 선택하여 대체

$$\rightarrow$$ AutoPrompt와 거의 동일하지만, 차이점: ***search over ALL possible tokens to replace at each step, rather than just a single one***

