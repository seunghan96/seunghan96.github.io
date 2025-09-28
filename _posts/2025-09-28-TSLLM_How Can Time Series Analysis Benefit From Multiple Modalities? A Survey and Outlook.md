---
title: How Can Time Series Analysis Benefit From Multiple Modalities? A Survey and Outlook
categories: [LLM, MULT, TS]
tags: []
excerpt: arxiv 2025
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# How Can Time Series Analysis Benefit From Multiple Modalities? A Survey and Outlook

- 인용 수 (2025-08-27): 9회

- https://arxiv.org/pdf/2503.11835

- https://github.com/AdityaLab/MM4TSA

<br>

# **1. Abstract**

Time Series Analysis (TSA)

- 여전히 **단일 modality** 중심으로 연구

- 언어·비전·오디오 등 **“rich modalities”**에 비해 상대적으로 고립됨.


<br>

### **MM4TSA (Multiple Modalities for TSA)**

TSA를 강화하기 위한 3가지 방식 제시:

- **TimeAsX**: 다른 modality의 foundation model 재사용
- **Time+X**: Multimodal 확장
- **Time2X / X2Time**: cross-modal 상호작용

<br>

분류

- modality 별 (text/image/audio/table)
- domain 별(finance/medical/spatial-temporal) 

<br>

# **2. Introduction**

## (1) **Motivation & Proposal**

T는 **다른 modality와의 통합 부족**!!

$$\rightarrow$$ Interpretability와 generalization 측면에서 한계가 있음.

<br>

반면 language, vision 등:

- Foundation models(GPT, ViT 등)의 급성장으로 발전 가속화!

<br>

# **3. Related Works**

기존 survey들:

- 대부분 **LLM을 TSA에 사용하는 방법(Time as Text)**에만 치우쳐 있음

<br>

MM4TSA

- Modality 전체 (text/image/audio/table) 및 interaction까지 포괄

- 핵심 분류 구조:

  - **TimeAsX**: 다른 modality foundation model 재사용
  - **Time+X**: multimodal 확장
  - **Time2X / X2Time**: input/output을 넘나드는 interaction

- Domain별 구조 (Finance, Medical, Spatial-temporal)
- modality 유형별 서브분류 (예: Text → Prompt, Caption, Retrieval 등) 

<br>

# **4. Methodology (Taxonomy)**

## (1) **TimeAsX: Foundation Model Reuse**

- **Text**: GPT 활용 (LLMTime, ChatTS 등), prompt 기반/embedding 기반/quantization 기반 alignment 방식
- **Image**: line-graph, heatmap, spectrogram, GAF 등으로 변환 → ViT 등 vision model 활용
- **Audio**: wavelet transform 등으로 변환 → AST/SSAST 활용
- **Table**: TabPFN 기반 구조 활용. Tabular feature로 TS 표현

<br>

## (2) **Time+X: Multimodal Fusion**

- Text를 **정적 (meta info) & 동적(news, weather 등)**으로 구분하여 **TS에 결합**

- Fusion 방식
  - **Early / Intermediate / Late**

- 예시: MM-TSFlib, GPT4MTS, TGForecaster, DualForecaster

- Domain별 Multimodal 구조:

  - Finance: 뉴스 + 가격
  - Medical: ECG + Report + Tabular + X-ray
  - Spatial: 센서값 + 이미지/뉴스/지도 등


<br>

## (3) **Time2X & X2Time: Cross-Modality Interaction**

- **Text2Time**: textual description 기반 synthetic TS 생성
- **Time2Text**: TS caption, 설명 자동 생성
- **양방향 QA**: ChatTime, Time-MQA 등
- 의료: ECG → report or QA, Text → synthetic ECG 등

<br>

# **5. Experiments**

### MM4TSA 주요 Dataset 

| **Dataset**   | **Modalities**    | **Domain**    | **특징**              |
| ------------- | ----------------- | ------------- | --------------------- |
| Time-MMD      | Time + Text       | General       | 9 domains, 24년 분량  |
| ChatTime      | Time + Text       | Weather 등    | 날짜/날씨 info 포함   |
| TSQA          | Time + Text       | Multi-task QA | human-curated 1.4k QA |
| MIMIC, PTB-XL | Time + Text/Image | Medical       | EHR + ECG + 보고서    |
| Terra         | Time + Text/Image | Climate       | 전 지구적 데이터셋    |

<br>

### **비교 구조/전략 요약**

- 다양한 fusion 방식 (Early/Late/Intermediate) 별로 주요 모델 사례 설명
- Text2Time의 경우 CLaSP, BRIDGE, ChatTS 등이 대표적
- Time2Text는 Captioning/Explanation/QA 등으로 나뉨 

<br>

# **6. Conclusion**

MM4TSA는 TS 분석을 위해 

- multimodal fusion
- foundation model reuse
- interaction

을 통합한 새로운 패러다임.