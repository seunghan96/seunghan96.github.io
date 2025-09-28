---
title: Time-RA: Towards Time Series Reasoning for Anomaly with LLM Feedback
categories: [LLM, MULT, TS]
tags: []
excerpt: KDD 2026
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Time-RA: Towards Time Series Reasoning for Anomaly with LLM Feedback (KDD 2026)

- 인용 수 (2025-08-27): 0회
- https://arxiv.org/pdf/2507.15066

<br>

## **1. Abstract**

기존 TSAD의 한계

- 단순 binary classification 분류에 집중
- **fine-grained anomaly 유형 구분 및 설명 능력은 부족**.

<br>

Proposal: 새로운 태스크 **Time-RA**

- TSAD를 **generative reasoning task**로 재정의

- 실제 TS를 기반으로 한 **RATs40K**라는 최초의 **multimodal reasoning dataset**도 구축




Experiments

- 여러 LLM 및 MLLM을 benchmarking하여 SFT의 중요성과 모델 한계점 분석.

- 코드 및 데이터셋 공개:

  - Code: [github.com/yyysjz1997/Time-RA](https://github.com/yyysjz1997/Time-RA)
  - Dataset: [huggingface.co/datasets/Time-RA/RATs40K](https://huggingface.co/datasets/Time-RA/RATs40K) 


<br>

![image-20250827104734807](/Users/seunghan96/Library/Application Support/typora-user-images/image-20250827104734807.png)

.

# **2. Introduction**

![image-20250827104759164](/Users/seunghan96/Library/Application Support/typora-user-images/image-20250827104759164.png).

## (1) **Motivation**

기존 TSAD 한계점

- 실제 TSAD는 단순 탐지 이상으로 **"이유와 원인"에 대한 해석**이 중요!

  $$\rightarrow$$ But 대부분 연구는 ***detection에만*** 집중.

- 기존 데이터셋은 ***anomaly type 미구분*** or ***synthetic*** 중심
- ***Text 및 Image modality 부족***

- LLM 및 MLLM은 reasoning 잠재력이 있으나, ***task-specific data 부재***로 한계.

<br>

## (2) **Proposal**

### **Time-RA**

- Input & Output
  - Input: **TS + text + image**
  - Output: **Anomaly 여부, 유형 분류, reasoning**


- **Structured prompting**으로 LLM fine-tuning

- Observation → Thought → Action 구조의 **human-style explanation pipeline** 설계 

<br>

# **3. Related Works**

(1) **TSAD**

- ARIMA, AE, LSTM 등에서 Transformer 기반까지 발전
- But 대부분 **binary** detection 중심

<br>

(2) **Multimodal LLM for TS**: 

- ChatTime, LLMAD 등 일부 존재
- **실제 데이터 다양성 부족**

<br>

(3) **Dataset**: 

- UCR, Yahoo, NAB, MSL, SMAP 등은 **single modality**
- 기존 multimodal은 대부분 synthetic 또는 domain 협소

<br>

# **4. Methodology**

## (1) Task 정의

- Input: TS ($$T$$), Text ($$D$$), Image ($$V$$)

- Output:

  - **Detection**: $$y_l = \pi_{\text{detect}}(T, D, V) \in \{0,1\}$$.
  - **Classification**: $$a = \pi_{\text{classify}}(T, D, V) \in C_{\text{uni}} \cup C_{\text{multi}}$$.
  - **Reasoning**: $$r = \pi_{\text{reason}}(T, D, V)$$.
  

<br>

## (2) **Anomaly Taxonomy**

- **Univariate: 14가지 유형** (e.g., Sudden Spike, Drift, Flatline, Pattern Change 등)

- **Multivariate: 6가지 유형** (e.g., Covariance Shift, Trend Divergence, Temporal Dependency 등)

  



![image-20250827105046355](/Users/seunghan96/Library/Application Support/typora-user-images/image-20250827105046355.png).

![image-20250827105053763](/Users/seunghan96/Library/Application Support/typora-user-images/image-20250827105053763.png).

.

<br>

## (2) **Prompt Design**

- Structured 3-step 구조: 
  - **Observation → Thought → Action**

- Prompt에 전문가 정의 anomaly 설명과 예시 포함
- LLM이 자동으로 Reason + Action 생성

<br>

# **5. Experiments**

## (1) **Dataset:** **RATs40K**

![image-20250827104828337](/Users/seunghan96/Library/Application Support/typora-user-images/image-20250827104828337.png).

- 총 **39,574 samples**

  - 10 domains (AIOps, Healthcare, Finance, IoT 등)
  - Modalities: Time + Text + Image
  - 14 (U) + 6 (M) anomaly types
  
- 모든 sample에 대해 **GPT-4 기반 ranking + feedback refinement 적용**

- Annotation 품질은 전문가 검토로 정량/정성 평가됨 

<br>

## (2) **Benchmark Models**

- **LLM**: Qwen2.5, Llama-3, DeepSeek, Phi-4 등
- **MLLM**: Llava-v1.5, Qwen2.5-VL
- **기존 모델**: XGBoost, LSTM, LOF, AE1SVM 등

<br>

## (3) **Tasks**

- Anomaly Detection (Label)
- Type Classification (Action)
- Reasoning Generation (Thought)

<br>

## (4) **평가 방법**

- Label/Action: Precision, Recall, F1
- Thought: Cosine, TF-IDF, Levenshtein, Token-based 유사도
- 추가로 human-in-the-loop 평가도 포함 (Likert 1~5 scale, Fig. 5) 

<br>

## (5) **주요 결과**

- SFT는 LLM 성능 전반 향상 (Qwen2.5 F1: 0.90+)
- MLLM은 시각화 정보 추가 시 Thought quality 개선
- Fine-tuned LLM은 기존 TSAD 모델보다 높은 generalization 확보 (zero-shot에서도 강건함)
- Multivariate보다 Univariate에서 reasoning precision 높음

<br>

## **6. Conclusion**

- **Time-RA**는 단순 detection을 넘은 **multimodal, fine-grained, reasoning 기반 TSAD의 새로운 방향성** 제시.

- RATs40K는 품질 높은 annotation과 다양한 도메인 커버리지를 갖춘 **최초의 대규모 실세계 멀티모달 TSAD 데이터셋**.

- 향후 연구 방향:

  - Multi-label anomaly
  - Domain adaptation
  - Long sequence modeling + visual 강화
  - Continual learning 기반 reasoning 개선 등 
  
  
