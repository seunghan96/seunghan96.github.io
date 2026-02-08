---
title: Multi-modal Time Series Analysis: A Tutorial and Survey
categories: [LLM, TS]
tags: []
excerpt: arxiv 2025
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Multi-modal Time Series Analysis: A Tutorial and Survey

https://arxiv.org/pdf/2503.13709

<br>

# **1. Introduction**

## (1) Overview

### a) Background

- **TS Analysis**: 다양한 도메인에서 핵심적인 역할
- 기존 연구: **temporal dynamics** 모델링에 집중

<br>

### b) Challenges

- 실제 환경: TS는 단독으로 존재 X
- 대부분 **external context**와 함께 관측됨
  - 금융: stock price + news text
  - 의료: physiological signals + clinical notes
  - 교통: traffic flow + textual / spatial context
- 이런 context는 **multi-modal** 형태 (Text, Image, Tabular, Graph 등)

$$\rightarrow$$ **Multi-modal TS의 필요성**: 더 **comprehensive view** 제공

<br>

## (2) **Challenges**

- (1) **Heterogeneity**: 각 modality는 통계적 특성과 구조가 다름
  - TS: temporal dependency 중심
  - Text/Image: semantic structure 중심
- (2) **Modality Gap**
- (3) **Temporal Misalignment**
  - modality마다 timestamp, granularity가 다름
- (4) **Contextual Noise**
  - 실제 데이터에는 task와 무관한 정보가 많음

<br>

## (3) **최근 연구 동향**

- DL 기반 **cross-modal interaction** 기법

- **기존 survey의 한계**
  - a) Task-specific, domain-specific
  - b) Unified perspective 부재

<br>

## (4) 이 논문의 목표

Multi-modal TS Analysis를 **systematic** + **unified** + **up-to-date** 정리

<br>

핵심 개념을 **cross-modal interaction framework**로 정리

- (1) Interaction type: 
  - **Fusion / Alignment / Transference**
- (2) Interaction stage: 
  - **Input / Intermediate / Output**

<br>

### Contribution

- 40개 이상의 대표적 multi-modal TS 방법 정리
- Unified taxonomy 제시
- 다양한 real-world application 정리
- Future research direction 제안

<br>

# **2. Background and Our Scope**

## (1) Multimodal ML

서로 다른 modality를 **jointly modeling**

- (1) **Representation learning**
  - Modality별 특성과 joint semantics를 동시에 encoding

- (2) **Cross-modal interaction**
  - Modality 간 element-level 관계 포착

- (3) **Knowledge transference**
  - 한 modality에서 학습한 정보 $$\rightarrow$$ 다른 modality로 전달
- (4) **Theoretical / empirical analysis**
  - Multi-modal learning의 성질 분석

<br>

## (2) Multi-modal TS Analysis

**이 논문의 관점**

- 단순한 modality 결합이 아니라, **cross-modal interaction**을 어떻게 설계하느냐!
- 세 가지 관점
  - **Data** (Section 3)
  - **Cross-modal Interaction Methods** (Section 4)
  - **Applications** (Section 5)

<br>

**Scope 명확화 1: TS 유형**

- 다루는 대상: **Standard TS** & **Spatial TS**
- Spatial structure (graph)
  - TS에 내재된 구조로 간주
  - 독립적인 modality로 취급하지 않음

<br>

**Scope 명확화 2: Multi-modal의 기준**

- 외부 real-world context를 활용하는 방법에 집중

- **의도적으로 제외한 접근**: TS를

  - image로 변환
  - table로 변환

  해서 **single-modality**로 처리하는 방법

<br>

**기존 survey와의 차별점**

- a) **Vision 중심** survey
  - Imaging-based TS transformation에 국한
- b) **LLM reasoning** 중심 survey
  - Multi-modal reasoning에 초점
- 본 논문: modality나 task에 제한되지 않고 **통합적 taxonomy + interaction 중심 분석** 제공

<br>

# **3. Multi-modal TS Data**

## (1) Data Sources and Modalities

Multi-modal TS

- **TS**를 중심 modality+ 다양한 **context modalities**를 함께 활용

<br>

[Main] **TS (Primary Modality)**

- (연속 또는 이산적인) Temporal signal
- 핵심 특성
  - strong **temporal dependency**
  - non-stationarity
  - noise sensitivity

<br>

[Sub] **Text Modality**

- 가장 널리 사용되는 auxiliary modality
- ex) news articles, clinical notes .. 
- 처리 방식: Transformer-based encoder, pre-trained LLM
- 역할
  - latent semantic context 제공
  - event-driven dynamics 설명

<br>

[Sub] **Image / Video Modality**

- 시각적 context 제공
- ex) satellite images (climate, traffic), medical images ... 
- 특성
  - Spatial information이 강함
  - TS와 temporal alignment 필요

<br>

[Sub] **Tabular / Metadata**

- Static or slowly-varying context
- ex) Demographics, device information ... 

<br>

[Sub] **Graph / Spatial Structure**

- Sensor network, traffic network 등

<br>

## (2) Temporal Alignment and Granularity

**Temporal Alignment 문제**: modality마다

- sampling rate

- timestamp

- observation frequency

  가 다름

<br>

**대표적 alignment 방식**

- (1) **Hard alignment**
  - Timestamp 기준으로 직접 매칭
  - Interpolation, aggregation 활용
- (2) **Soft alignment**
  - Attention 기반 alignment
  - Latent time mapping 학습
- (3) **Event-based alignment**
  - 특정 event 발생 시점을 기준으로 결합

<br>

**Granularity mismatch**

- ex) minute-level sensor + daily news
- 해결 전략
  - up/down sampling
  - hierarchical temporal modeling

<br>

## (3) Data Challenges

- (1) **Missing Modality**
  - 일부 timestamp에서 특정 modality가 존재하지 않음
- (2) **Noise and Irrelevance**
  - auxiliary modality가 항상 유용하지 않음
  - task-irrelevant information 포함 가능
- (3) **Scalability**
  - multi-modal 데이터는 storage, computation 비용 증가
- (4) **Label Scarcity**
  - multi-modal dataset은 annotation 비용이 큼

<br>

## (4) Representative Datasets

Examples)

- **Healthcare**
  - physiological signals + clinical text
- **Finance**
  - market TS + news / social media
- **Transportation**
  - traffic sensor + spatial / visual data
- **Climate**
  - meteorological TS + satellite imagery

<br>

**중요한 관찰**: 대부분의 dataset은

- 특정 task에 강하게 종속됨

- general-purpose benchmark가 부족함

<br>

# **4. Cross-modal Interaction Methods**

두 개의 축

- (1) **interaction 방식**
- (2) **interaction 시점**

<br>

## (1) Interaction Taxonomy 개요

- 모든 방법은 다음 두 질문으로 분류 가능
  - **What to interact**: modality 간에 무엇을 주고받는가
  - **When to interact**: 모델의 어느 stage에서 상호작용하는가
- 이를 통해 제안하는 unified taxonomy
  - **Interaction Type**
    - Fusion
    - Alignment
    - Transference
  - **Interaction Stage**
    - Input-level
    - Intermediate-level
    - Output-level

<br>

## (2) Interaction Types

### a) Fusion

- **정의**: 여러 modality의 representation을 하나의 joint representation으로 **통합**
- **대표적 방식**
  - Concatenation
  - Summation
  - Gated fusion
  - Cross-attention
- **특징**
  - 구조가 단순
  - end-to-end 학습이 쉬움
  - modality 간 관계를 명시적으로 제어하기 어려움
- **TS 관점**
  - auxiliary modality는 TS representation을 보완하는 역할
  - early fusion vs late fusion 차이가 큼

<br>

### b) Alignment

- **정의**: modality 간 **correspondence**를 명시적으로 학습
  - time axis 또는 semantic space에서 정렬
- **대표적 기법**
  - Cross-modal attention
  - Contrastive learning
  - Dynamic time alignment
- **핵심 아이디어**
  - “어느 text가 어느 time step과 관련 있는가”
  - “어느 image patch가 어느 temporal pattern과 대응되는가”
- **장점**
  - interpretability 향상
  - noisy modality에 강함

<br>

### c) Transference

- **정의**: 한 modality의 knowledge를 다른 modality로 **전이(transfer)**
- **대표적 방식**
  - Teacher–Student learning
  - Representation distillation
  - Auxiliary task learning
- **사용 시점**
  - inference 시 auxiliary modality가 없을 때
  - missing modality 문제 해결
- **TS에서의 의미**
  - training 단계에서만 text/image 사용
  - inference에서는 TS 단독 사용 가능

<br>

## (3) Interaction Stages

### a) Input-level Interaction

- **개념**: raw input 또는 shallow embedding 단계에서 결합
- **예시**
  - TS embedding + text embedding concat
  - positional alignment 후 joint input 구성
- **장점**
  - 구현 간단
  - modality 정보가 early부터 반영됨
- **한계**
  - heterogeneity handling이 어려움
  - noise propagation 위험

<br>

### b) Intermediate-level Interaction

- **개념**: modality별 encoder 이후 hidden representation 단계에서 상호작용
- **대표적 구조**
  - dual-encoder + cross-attention
  - shared latent space projection
- **장점**
  - modality 특성을 유지한 채 interaction 가능
  - 가장 널리 사용되는 방식
- **논문의 관찰**: 성능과 안정성 측면에서 가장 **robust한 선택**

<br>

### c) Output-level Interaction

- **개념**: modality별 prediction을 결합
- **예시**: ensemble, decision-level fusion
- **특징**
  - 모델 간 독립성 유지
  - deep interaction은 제한적

<br>

## (4) Interaction Design Considerations

**Task dependency**

- a) Forecasting: temporal alignment 중요
- b) Classification: semantic fusion 중요

<br>

**Data availability**

- a) missing modality 여부
- b) inference-time constraint

<br>

**Model complexity**

- a) alignment-based methods는 계산량 증가

<br>

## (5) **핵심 takeaway**

- 성능 차이는
  - modality 자체보다
  - **interaction design**에서 발생
- 특히 아래가 중요!
  - **Intermediate-level + Alignment/Fusion**
  - **Transference for missing modality**