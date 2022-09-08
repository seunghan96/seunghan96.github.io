---
title: Contrastive Learning 소개 (2)
categories: [CL]
tags: []
excerpt: Contrastive Learning
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( 참고 : https://velog.io/@sjinu/Similarity-Learning-Contrastive-Learning#:~:text=%EB%8C%80%EC%A1%B0%EC%A0%81%20%ED%95%99%EC%8A%B5(Contrastive%20Learning)%EC%9D%98,representation)%EC%9D%84%20%ED%95%99%EC%8A%B5%ED%95%98%EB%8A%94%20%EA%B2%83%EC%9D%B4%EB%8B%A4. )



# 1. Similarity Learning (SL)

output : 유사도

model의 핵심

- 비슷한건 **높은 유사도**
- 상이한건 **낮은 유사도**

<br>

## (1) Regression SL

Data : 2개의 데이터(x)에 대한 **유사도 (y label)**가 이미 주어져 있음

- 한계점 : 현실에서, 두 데이터의 유사도를 어떻게 정의..?

<br>

## (2) Classification SL

Data : 2개의 데이터(x)에 대한 **유사 여부 (y label)**가 1/0 값으로 이미 주어져 있음

- 한계점 : "얼마나" 유사한지에 대한 정도 X

<br>

## (3) Ranking SL

Data : $$\left(x_{i}, x_{i}^{+}, x_{i}^{-}\right)$$

Model : $$f\left(x, x^{+}\right)>f\left(x, x^{-}\right)$$를 보장하도록!

장점

- 사전에 y label (유사도 정보)가 필요없다.
- **Contrastive Learning (대조적 학습)** 이라고도 함

<br>

### Contrastive Learning

***비슷한건 가깝게, 다른건 멀게!***

( = Distance Metric Learning )

<br>

# 2. Distance Metric Learning

## (1) Metric Learning

데이터들 사이의 **distance function**을 학습하는 방법론

- metric = distance ( function )

<br>

Metric이 되기 위한 조건

- (1) Non-negativity: $$f(x, y) \geq 0$$

- (2) Identity of Discernible: $$f(x, y)=0 \Leftrightarrow x=y$$
- (3) Symmetry: $$f(x, y)=f(y, x)$$
- (4) Triangle Inequality: $$f(x, z) \leq f(x, y)+f(y, z)$$

<br>

## (2) 두 종류의 Metric

### a) Pre-defined metric

- 학습 (X)
- ex) Euclidean, Manhattan distance..

<br>

### b) Learned Metric

- 학습 (O)
- Ex) Mahalanobis : $$f(x,y) = (x-y)^T M (x-y)$$
  - $$M$$ : 주어진 데이터로부터 계산된 행렬

<br>

## (3) Deep Metric Learning

딥러닝을 사용해서 푸는 많은 데이터들은 **매우 고차원**

- ex) Image, Text, Speech ... 

$$\rightarrow$$ 거리 비교가 쉽지/직관적이지 않음!

<br>

따라서, 해당 고차원의 이미지를 non-linear projection하여 **저차원의 manifold**를 찾아야!

$$\rightarrow$$ NN을 사용해서!

<br>

### ex) Siamese CNN (샴 CNN)

Input : 2개의 이미지

Output : 2개의 이미지 사이의 거리

<br>

### ex) Triplet Network ( Ranking Similarity )

Input : 3개의 이미지

- 1개의 이미지 : Query Image ( 알고자 하는 이미지 )
- 2개의 이미지 :
  - (1) POSITIVE 이미지 ( Query Image와 유사 )
  - (2) NEGATIVE 이미지 ( Query Image와 상이 )

<br>

# 3. Loss Function : Contrastive Loss

- Loss : $$\sum$$ (loss of POSITIVE pairs) + $$\sum$$ (loss of NEGATIVE pairs)

  - ex) loss of POSITIVE pairs

    - $$L\left(x_{p}, x_{q}\right)=\mid \mid x_{p}-x_{q} \mid \mid ^{2}$$..... 2개가 서로 **가까워 지도록**

  - ex) loss of NEGATIVE pairs

    - $$L\left(x_{n}, x_{q}\right)=\max \left(0, m^{2}-\mid \mid x_{n}-x_{q}\mid \mid^{2}\right)$$ .... 2개가 서로 **멀어 지도록**

      ( = Hinge Loss , where $$m$$ = margin )

    - 의미 : margin 이상 만큼 차이가 난다면, 손실 부여 X

- 정리 : $$\operatorname{Loss}\left(x_{p}, x_{q}, y\right)=y *\mid \mid x_{p}-x_{q}\mid \mid^{2}+(1-y) * \max \left(0, m^{2}-\mid \mid x_{p}-x_{q}\mid \mid^{2}\right)$$.

  - $$y$$는 positive / negative pair의 label
    - positive pair : $$y=1$$
    - negative pair : $$y=0$$

<br>

***Positive Margin***

- Positive Pair이어도, 너무 같은 공간에 놓이는 것을 방지하기 위해 positive margin 부여 가능!
  - (PM (O)) $$\left[d_{p}-m_{\text {pos }}\right]_{+}+\left[m_{n e g}-d_{n}\right]_{+}$$
  - (PM (X))  $$d_{p} +\left[m_{n e g}-d_{n}\right]_{+}$$.

<br>

# 4. 요약

Contrastive Learning의 핵심 :

- (1) model (DL)
- (2) metric
  - ex) cosine similarity, euclidean distance ...

