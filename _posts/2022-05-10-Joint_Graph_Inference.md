---
title: (paper) A Study of Joint Graph Inference and Forecasting
categories: [GNN, TS]
tags: []
excerpt: Graph Neural Network (2021)

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# A Study of Joint Graph Inference and Forecasting (2021)

# 0. Abstract

GNN MTS 논문 비교

4개의 모델 사용 : 

- (1) GTS ( Graph for Time Series )
- (2) GDN ( Graph Deviation Network )
- (3) MTGNN ( MTS forecasting with GNNs )
- (4) NRI ( Neural Relational Inference )

<br>

# 1. Introduction

Recent Works는 다음에 집중

$$\rightarrow$$ ***JOINTLY*** inferring the “relation” between TS & “forecast” e2e

<br>

아래의 3개의 research question에 집중

- [R1] graph inference & forecasting 방법론이 어떻게 성능 개선에 기여하는지

- [R2] 성능 비교

- [R3] 추론된 graph의 특징 비교

  ( 얼마나 일관되게 생성되었는지? 실제 정답 그래프 구조와 얼마나 차이가 있는지? )

<br>

# 2. Background

그래프 구조 (Graph Structure)의 3가지 경우

- (1) known
- (2) unknown
- (3) partially known

<br>

# 3. Literature Review

모든 모델은 다음의 2가지 구조로 구성됨

- (1) GRAPH LEARNING : $$A$$ 학습
- (2) FORECASTING MODULE : TS 예측

<br>

## (1) Graph Learning

대부분, $$A$$ 가 sparse하기를 원함

( $$\because$$ 너무 많은 노드들로부터 정보 취합 X )

<br>

### (a) Adjacency matrix 학습 방식 차이

대부분, **노드 간 pairwise score**를 구함으로써 graph learning

- MTGNN & GDN : top $$K$$ 고르기

  - 장점 : sparse함 강제 유지
  - 단점 : 미분 불가 ( 완전히 e2e 되진 X )

- NRI & GTS : pairwise score [0,1]로 보낸 뒤, Gumbel softmax trick으로 샘플링

  ( https://seunghan96.github.io/ml/stat/Gumbel_Softmax_Trick/ 참고 )

<br>

### (b) node 별 hidden representation 구하는 방식의 차이

- MTGNN & GDN : **h = 노드 임베딩 ( …. TS값과 무관 )**

- GTS & NRI : **h = ENC( raw TS )**
  - GTS : global ( = 위의 ENC가 shared ) & 전체 TS 통째로 사용
    - 장점 : less-flexible
    - 단점 : cheap ( 전체 TS사용해서 학습하므로, batch별로 adjacency matrix가 다르지 X )
  - NRI : window 별로 representation 생성
    - 장점 : flexible
    - 단점 : expensive ( batch별로 adjacency matrix 다 가지고 있어야 )

<br>

## (2) Graph-based Forecasting

- MTGNN : TCN & GCN을 interchange
- GTS : DCRNN 사용
  - 매 노드의 hidden state는 매 타임스텝마다 diffuse

핵심 : **(1) Graph Learning**을 통해 얻어낸 graph structure는 “미분가능한” 방식으로 **(2) forecasting**에 반영되어야!

( 그래야, 학습되는 graph structure가 forecasting에 도움되는 방식으로 적응(adjust)됨 )

<br>

# 4. Experiment

데이터 : real-world & synthetic

- real-world : **METRA-LA** & **PEMS-BAY**
- synthetic : **Diffusion** & **DAG**

<br>

Graph Structure

- (1) Ground-truth
- (2) Random Graph
- (3) No Graph

<br>

Scaling : 

- $$N(0,1)$$ : 대부분

- $$\text{MinMax}[0,1]$$ : SWaT, WADI 데이터셋

( 참고 : metric ( MAE ) 계산 시, original scale로 다시 가져와서 계산 )

<br>

### 기타 사항

- NRI : METR_LA에서 못함

  - 이유 1 : 사실 METR_LA의 구조는 static함
  - 이유 2 : METR_LA의 노드 개수 많아

  ( NRI는 per-window 단위로 예측하므로 ) 

<br>

- GNN > 비-GNN
  - datawet with an underlying spatial graph
    - ex) Traffic, Electricity, Solar Energy

<br>

- GTS에서, “true” graph를 사용하는것은 오히려 성능 drop ( 1개 데이터셋 빼고)
  - 즉, **graph learning**은 매우 중요!
- GTS’s forecasting module : **sparsity of graph** 덕을 많이 봄

<br>

- GDN : 노드 임베딩을 graph learning & forecasting 모두에 사용함

<br>

- 실제 ground-truth 구조와는 그닥 비슷 X

  “예측에 도움”이 되는 방향으로 graph learningd이 진행됨.

  

  