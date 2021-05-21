---
title: \[multimodal\] Neural FE of signal data - (1) Wav2Vec
categories: [MULT]
tags: [Multimodal Learning]
excerpt: Signal Data, Wav2Vec, SincNet, PASE
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# [ Neural Feature Extraction of signal data ]

기존의 (NN을 사용하지 않은) Feature Extraction 방법은 주로 "지식/공식"에 기반한 deterministic한 형태의 추출 방법이었다면, 이번에 다룰 Neural Featrue Extraction은 "특정 목적을 수행하기 위해 적절한" feature를 뽑아내기 위한 방법이다. (즉, task에 따라 같은 data에서도 feature가 다르게 뽑힐 수 있는 non-deterministic한 방법이다 )

Neural Feature Extraction의 대표적인 2가지 방법은 아래와 같다.

- 1) Wav2Vec
- 2) SincNet ( + PASE )

이번 포스트에서는 **Wav2Vec**에 대해서 다룰 것이다.

<br>

# 1. Wav2Vec

### (1) 구성 

( 둘 다 CNN )

- 1) encoder network $$f$$
  - $$X \rightarrow Z$$.
- 2) context network $$g$$
  - $$Z \rightarrow C$$.
  - 여기서 변환된 $$C$$를 feature로써 사용!

<br>

<img src= "https://i.imgur.com/H9X1HiX.png" width="400" />.

<br>

### (2) 학습 과정 

- **binary classification **

  ( 입력으로 들어오는 sample 쌍이 positive인지 negative인지 )

- positive ) $$C_i$$ & $$Z_{i+1}$$

  negative ) $$C_i$$ & $$Z_{i+1}$$ 외의 다른 random sample

- 학습 과정에서,

  - positive sample은 서로 가까워지도록
  - negative sample은 서로 멀어지도록

  학습이 이루어진다.

<br>

# 2. VQ-Wav2Vec

Wav2Vec + Vector Quantization

### (1) 구성 

( 둘 다 CNN )

- 1) encoder network $$f$$ ( Wav2Vec과 동일 )
- **2) Vector Quantization 모듈**
  - 자세한 내용은 뒤에서
- 3) context network $$g$$ ( Wav2Vec과 동일 )

<img src= "https://i.imgur.com/ivviYL1.png" width="400" />,.

<br>

### Vector Quantization 모듈

- continuous $$Z$$ $$\rightarrow$$ discrete $$\hat{Z}$$
- 대표적 방법 : Gumbel Softmax & K-means Clustering

<br>

**a) Gumbel Softmax**

- *Gumbel Softmax에 관한 구체적인 내용은이전에 올렸던 포스트 참조!*

  ( https://seunghan96.github.io/ml/stat/Gumbel_Softmax_Trick/ )

- Step 간단 요약

  - 1) logit으로 변환 ( + Linear Transformation )
  - 2) Gumbel Softmax
  - 3) argmax
  - 4) Embedding matrix와 내적

<img src= "https://i.imgur.com/y15Qu5Z.png" width="400" />.

<br>

**b) K-means Clustering**

- $$Z$$와 embedding matrix의 벡터와의 거리 (euclidean distance) 계산
- 가장 가까운 embedding matrix의 벡터 선택

<img src= "https://i.imgur.com/nrY2IAx.png" width="400" />.

<br>

# Reference

https://ratsgo.github.io/speechbook/docs/neuralfe/wav2vec

