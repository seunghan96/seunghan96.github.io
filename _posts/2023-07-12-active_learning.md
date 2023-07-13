---
title: Active Learning
categories: [ML]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# [ Active Learning ]

### Motivation of Active Learning

잠재적으로 뛰어난 기계를 두고, 사람이 모든 라벨링을 진행??

어떤 데이터가 필요한지를 기계가 판단 & 사람에게 라벨링을 부탁하는건?

$$\rightarrow$$ Active Learning

 

# 1. Active Learning

기계가 라벨링이 필요한 데이터 중 자동적으로, 그리고 점진적으로 가장 정보량이 많은 데이터를 선택하는 것을 목표로 한다. 

- step 1) 초기 라벨링된 일부 데이터를 이용해 모델은 학습을 시작

- step2 ) 라벨링이 이루어지지 않은 데이터 중 학습에 중요한 것들에 대해 라벨링을 요구

<br>

# 2. Active Learning Scenario

( 참고: Learner = 모델 )

 ## 1) Membership Query Synthesis

Learner가 데이터를 생성 & 이에 대한 label을 인간에게 요청

<br>

## 2) Stream-Based Selective Sampling

들어오는 개별 Unlabeled 데이터들을 (정보량에 따라) labeling 가치 여부를 판단!

( query strategy를 통해 평가 )

- option 1) require label
- option 2) discard

<br>

## 3) Pool-based Sampling

(Unlabeled) Data pool에서 정보량을 측정

그 중, 정보량이 많은 data들을 선택!

<br>

# 3. Query Strategy

***Q) Which data instance to label??***

<br>

### Example

Output value ( after softmax )

$$\begin{array}{|c|c|c|c|}
\hline \text { Data } & \text { class A } & \text { class B } & \text { class C } \\
\hline \mathrm{d} 1 & 0.9 & 0.09 & 0.01 \\
\hline \mathrm{d} 2 & 0.2 & 0.5 & 0.3 \\
\hline
\end{array}$$

<br>

## 1) Least Confidence (LC)

- $$d1$$ = 0.9
- $$d2$$ = 0.5

$$\rightarrow$$ ask label of $$d2$$

<br>

## 2) Margin Samping

- $$d1$$ = 0.9-0.009 = 0.81
- $$d2$$ = 0.5-0.3 = 0.2

$$\rightarrow$$ ask label of $$d2$$

<br>

### 3) Entropy Sampling

Entropy ( = Uncertainty )

$$\mathrm{H}(X)=-\sum_{i=1}^n \mathrm{P}\left(x_i\right) \log \mathrm{P}\left(x_i\right)$$.

- $$d1$$ = 0.115
- $$d2$$ = 0.447

$$\rightarrow$$ ask label of $$d2$$

<br>

# 4. Summary

Overall procedure of **Active Learning**

- Step 1) Data Collection ( = Unlabeled )

- Step 2) Label some data
  - (1) Labeled data : $$D_L$$
  - (2) Unlabeled data : $$D_U$$

- Step 3) Train the model
  - using $$D_L$$

- ***Step 4) Select some unlabeled data to be labeled***
- Step 5) Repeat Step 3&4 ( until threshold )

<br>

# Reference

https://littlefoxdiary.tistory.com/52