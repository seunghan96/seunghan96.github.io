---
title: (paper) Attention-based LSTM for Aspect-level Sentiment Classification (2016)
categories: [NLP,HBERT]
tags: [NLP, HBM]
excerpt: Attention-based LSTM for Aspect-level Sentiment Classification (2016)
---

# Attention-based LSTM for Aspect-level Sentiment Classification (2016)

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

### Contents

0. Abstract
1. Contribution
2. Related Works
   1. Sentiment Classification at Aspect Level
   2. Sentiment Classification with NN
3. Attention-based LSTM with Aspect Embedding
   1. LSTM
   2. AE-LSTM
   3. AT-LSTM
   4. ATAE-LSTM

<br>

# 0. Abstract

문장의 감정은 단지 문장의 content 뿐만 아니라, 문장 내의 aspect와도 밀접!

따라서, 문장의 내용과 aspect 사이의 connection을 찾는게 중요함!

- ex) *"짜장면은 맛있었는데 서비스는 별로였어. 그래도 전체적으로는 괜찮아!"*
  - 음식 : Good
  - 서비스 : Bad

그러기 위해, 이 논문에서는 ***Attention-based LSTM for aspect-level sentiment classification***을 제안함

<br>

# 1. Contribution

- 1) Attention-based LSTM for aspect-level sentiment classification를 제안
- 2) Aspect information을 반영할 수 있는 2가지 방법을 제안
  - 방법 1) **concatenate aspect vector** into **"sentence hidden representation"**
  - 방법 2) additionally **append the aspect vector** into the **"input word"**

<br>

# 2. Related Work

## 2-1) Sentiment Classification at Aspect Level

대부분의 현재(현재...?2016...) approach의 문제점

- ***aspect 고려 없이***, 전체 문장에 대한 감정만 찾으려함

$$\rightarrow$$ 이를 풀기위해, manually **set of features**를 디자인함

이렇게 디자인된 feature들을 사용하여 sentiment classifier들이 많이 나오긴 하였으나...

- 문제점 1) **highly dependent on quality of features**
- 문제점 2) feature engineering is **labor intensive**

<br>

## 2-2) Sentiment Classification with NN

- 생략

<br>

# 3. Attention-based LSTM with Aspect Embedding

## 3-1) LSTM

- 생략 ( 그림만 참조 )

![figure2](/assets/img/nlp/nlp41.png)

<br>

## 3-2) AE-LSTM

**LSTM with Aspect Embedding**

- 취지 : 각각의 aspect에 대한 **embedding vector**를 학습하자!
- notation
  - aspect $$i$$의 embedding vector : $$v_{a_{i}} \in \mathbb{R}^{d_{a}}$$
  - aspect embedding들 matrix : $$A \in \mathbb{R}^{d_{a} \times \mid A \mid }$$

<br>

## 3-3) AT-LSTM

**Attention-based LSTM**

- 기존의 LSTM은, sentiment classification에 있어서 **어떠한 부분이 중요한지 캐치 X**

- Notation :

  - LSTM이 만들어낸 hidden vector들의 matrix : $$H \in \mathbb{R}^{d \times N}$$

    ( 구성 요소 : $$\left[h_{1}, \ldots, h_{N}\right]$$ )

  - aspect $$i$$의 embedding vector : $$v_{a_{i}} \in \mathbb{R}^{d_{a}}$$

  - 1로 이루어진 vector : $$e_{N} \in \mathbb{R}^{N}$$

- **attention weight vector $$\alpha$$** 구하기!

- 과정 )

  $$\begin{aligned}
  &M=\tanh \left(\left[\begin{array}{c}
  W_{h} H \\
  W_{v} v_{a} \otimes e_{N}
  \end{array}\right]\right) \\
  &\alpha=\operatorname{softmax}\left(w^{T} M\right) \\
  &r=H \alpha^{T}
  \end{aligned}$$.

- final sentence representation :  $$h^{*}=\tanh \left(W_{p} r+W_{x} h_{N}\right)$$
- final prediction : $$y=\operatorname{softmax}\left(W_{s} h^{*}+b_{s}\right)$$

![figure2](/assets/img/nlp/nlp42.png)

<br>

## 3-4) ATAE-LSTM

**Attention-based LSTM with Aspect Embedding**

- 직관적으로 알 수 있듯, AT와 AE 둘 다 사용한 것

  즉, ***aspect embedding이 attention weight를 계산하는데에 사용됨***

- input vector에 aspect embedding을 추가한다! ( 아래 그림과 같이 )

![figure2](/assets/img/nlp/nlp43.png)