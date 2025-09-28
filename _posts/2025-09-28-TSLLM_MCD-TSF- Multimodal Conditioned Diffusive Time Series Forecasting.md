---
title: MCD-TSF; Multimodal Conditioned Diffusive Time Series Forecasting
categories: [LLM, MULT, TS]
tags: []
excerpt: arxiv 2025
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# MCD-TSF: Multimodal Conditioned Diffusive Time Series Forecasting (arxiv 2025)

- 인용 수 (2025-08-27 기준): 3회
- https://arxiv.org/pdf/2504.19669

<br>

# **1. Abstract**

기존 Diffusion 기반 TSF

- 대부분 **numerical TS만 처리** 
- **Multimodal 정보 활용 부족**

<br>

Proposal: **"MCD-TSF"**

- Diffusion 기반 예측 모델

- **Timestamp + text**를 함께 조건으로 활용!
- **Classifier-Free Guidance (CFG)**를 사용하여 text 사용 강도를 조절

<br>

Experiments: 

- 8개 real-world 도메인에서 **state-of-the-art** 성능 입증 .

- Code: https://github.com/synlp/MCD-TSF

<br>

# **2. Introduction**

## (1) Motivation

Limitation of previous works

- (Deterministic) TS는 **무작위성 & 불확실성**을 포함 → ***Deterministic 모델로는 한계***
- (LLM) 최근 LLM 기반 접근은 TS을 text로 변환하는데, 이 과정에서 **수치 정보 손실** 발생.
- (Diffusion) 기존 diffusion 기반 TS 모델은 **단일 modality에 국한**
  - e.g., Timestamp/text 등 부가 정보는 활용 부족.


<br>

## (2) **Proposal: MCD-TSF**

**Timestamp와 Text** 정보를 함께 사용하는 **diffusion 모델**

- Timestamp는 temporal 구조 보완
- Text는 역사적 의미 강화

- Classifier free guidance (CFG)로 Text 정보 활용 정도를 **inference 시 동적으로 조절 가능!**

<br>

# **3. Related Works**

## (1) **Diffusion for Time Series**

예시: D3VAE, MG-TSD, CSDI

한계점

- (1) TSF/imputation 용도로 ***제한적 사용***

- (2) 대부분 temporal hierarchy나 ***external modality 활용에는 제한적***

<br>

## (2) **Multimodal TSF**

예시: TimeLLM, MM-TSF, PromptCast 등.

- 방식) Text embedding + TS input을 concat
- 단점) 
  - **LLM cost 과다**
  - Text 과의존성 문제
  - Timestamp 활용은 deterministic 처리만 존재 .

<br>

# **4. Methodology**

## (1) Architecture

![image-20250827144313588](/Users/seunghan96/Library/Application Support/typora-user-images/image-20250827144313588.png)

- 3가지 입력: 
  - TS $$X$$

  - Timestamp $$T$$

  - Text $$E$$

- **Diffusion process**를 통해 "Noise→ Future TS"

- 3가지 주요 구성:
  - **Multimodal Encoder**: Conv + BERT + timestamp 추출기
  - **Fusion Module**: TAA (timestamp), TTF (text)
  - **CFG**: text 사용 강도 $$w$$로 조절


<br>

## (2) **Training**

Forward & Reverse

- [Forward] Gaussian noise 추가
- [Reverse] $$Y_{k-1} \sim \mathcal{N}(\mu, \sigma^2 I)$$
  - Condition: $$(X, T, E)$$

- Loss: Conditional MSE (text 포함/미포함 모두 학습)

<br>

# **5. Experiments**

## (1) **Dataset**

- **Time-MMD** 기반 8개 도메인 
  - Agri, Climate, Economy, Energy, Env, Health, Social, Traffic

- Text coverage는 도메인별 4.2%~100%
- Timestamps: day/week/month → normalized vector
- Text: 과거 36기간의 보고서를 GPT 스타일 템플릿으로 구성 

<br>

## (2) **Task**

- Time Series Forecasting (short/medium/long)

<br>

## (3) **Baselines**

- **DIFF 계열**: DIFF, DIFF+TAA, DIFF+TTF, DIFF+TAA-T, DIFF+TTF-T
- **Transformer 계열**: PatchTST, Autoformer, FEDformer, Reformer, HCAN
- **MLP 계열**: FiLM, DLinear, TimeMixer++
- **LLM/Text**: TimeLLM, FPT, MM-TSF
- **Probabilistic**: CSDI, D3VAE
- **Timestamp**: TimeLinear, GLAFF

<br>

## (4) **Resource**

- BERT base 사용 (frozen)
- Fusion layer: TAA + TTF 6개 층
- DDIM 기반 noise schedule
- 학습 시 puncond=0.1, test 시 w=0.8 (CFG Text 사용 조절)
- Evaluation: MSE, MAE 

<br>

## (5) Results

- Timestamp-Assisted Attention (TAA)
- Text-Time series Fusion (TTF)

![image-20250827170209514](/Users/seunghan96/Library/Application Support/typora-user-images/image-20250827170209514.png).

![image-20250827170219915](/Users/seunghan96/Library/Application Support/typora-user-images/image-20250827170219915.png).

<br>

# **6. Conclusion**

**MCD-TSF** 

- **TS + timestamp + text**를 함께 통합
- 예측 정확도와 확장성을 개선한 diffusion 기반 모델

Experiments

- 8개 도메인 전반에서 기존 diffusion, transformer, LLM 기반 모델 모두에 비해 우수한 성능.

- **timestamp와 text의 동적 결합** + **CFG 조절 기법**이 핵심 기여.