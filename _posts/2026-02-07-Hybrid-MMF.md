---
title: Multi-Modal Forecaster; Jointly Predicting Time Series and Textual Data
categories: [LLM, TS]
tags: []
excerpt: arxiv 2024

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Multi-Modal Forecaster: Jointly Predicting Time Series and Textual Data

https://arxiv.org/pdf/2411.06735

<br>

# 0. Abstract

**Motivation**

- 기존 forecasting 방법들은 대부분 **unimodal**

  $$\rightarrow$$ TS와 함께 존재하는 **textual data**를 활용하지 못함!

- 주요 원인: **Well-curated multimodal benchmark dataset의 부재**

<br>

**Dataset (TimeText Corpus, TTC)**

- **Multimodal** forecasting dataset: **Time-aligned text + TS**로 구성
- Timestamp 기준으로, **numerical sequence와 text sequence**가 정렬됨
- 도메인: Climate science & healthcare

<br>

**Model (Hybrid Multi-Modal Forecaster, Hybrid-MMF)**

- **Multimodal LLM** 기반 모델
- **Shared embeddings**를 사용하여
  - TS forecasting & text forecasting을 **jointly** 수행

<br>

**Key Result**: Hybrid-MMF는 **기존 baseline 대비 성능 향상을 달성하지 못함**

**Implication**

- Multimodal forecasting은 직관과 달리 **"단순 결합으로 성능 개선이 어려움"**
- TS–text joint modeling의 **본질적 난이도와 한계**를 드러내는 **negative result**

<br>

# 1. Introduction

## (1) Background

### a) **Motivation**

- 기존 TS forecasting은 대부분 **unimodal**로 TS만 사용
- But 실제 환경에서는 **text meta-data**가 풍부하게 존재
- LLM은 text 이해에 강점이 있으며
  - TS의 **context 제공**
  - 수치 예측과 함께 **textual interpretation 생성** 가능

<br>

### b) **Challenge**

- TS와 text는 구조와 표현 방식이 **본질적으로 상이**
- 두 modality를 **동시에 encoding 및 forecasting**하는 것은 어려움

<br>

### c) Limitation

- 대규모, 정제된 **paired TS–text forecasting dataset 부재**

- Time-MMD?

  - Text prediction 관점에서는 여전히 한계 존재

- 기존 multimodal 접근들은

  - TS만 예측하거나

  - TS를 text 형태로 변환하는 **LLM-centric framing**에 집중

    $$\rightarrow$$ 진정한 **joint multimodal forecasting**은 아님

<br>

### d) **Problem Definition**

- a) TS forecasting

- b) cextual event forecasting

  $$\rightarrow$$**동시에 수행하는 joint multimodal forecasting**에 초점

<br>

## (2) Proposal

### a) **Dataset (TimeText Corpus, TTC)**

- **Timestamp-aligned TS + text**로 구성
- Domain
  - Climate science
  - Healthcare
- Multimodal forecasting을 위한 **benchmark dataset** 제공

<br>

### b) Modeling

- TS & Text를 **shared embedding**으로 통합

<br>

## (3) Contributions

- **Simultaneous Encoding of Multimodal Data**
  - TS와 text를 공동 표현 공간에서 encoding하는 방식 제안
- **Multimodal Dataset Contribution**
  - Mimic-III 기반 의료 데이터
  - National Weather Service 기반 기후 데이터
  - Multimodal TS forecasting benchmark로 공개

- **Key Observation**

  - Hybrid-MMF는 **competitive performance**를 보이나, baseline 대비 **개선 폭은 제한적**

    $$\rightarrow$$ Multimodal forecasting의 **난이도**를 실증적으로 보여줌!

  <br>

