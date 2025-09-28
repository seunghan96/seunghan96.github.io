---
title: ITFormerl Bridging Time Series and Natural Language for Multi-Modal QA with Large-Scale Multitask Dataset
categories: [LLM, MULT, TS]
tags: []
excerpt: ICML 2025
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# ITFormer: Bridging Time Series and Natural Language for Multi-Modal QA with Large-Scale Multitask Dataset (ICML 2025)

- 인용 수 (2025-08-27 기준): 0회

- https://arxiv.org/pdf/2506.20093

<br>

# **1. Abstract**

- [Task] **Time-Series QA**라는 새로운 Task 정의
- [Dataset] **Large-scale Multi-task TSQA 데이터셋 (EngineMT-QA)** 제안.
- [Model] **ITFormer**를 통해 TS encoder와 frozen LLM 간의 시맨틱 정렬 및 질의응답을 수행
  - 1% 미만 파라미터 추가만으로도 기존 SOTA 대비 뛰어난 QA 성능 달성.

- 코드 및 데이터 공개: https://pandalin98.github.io/itformer_site/ 

<br>

# **2. Introduction**

## (1) **Motivation**

- 산업/의료/기후 등에서 TS의 중요성 증가.
- 사용자는 자연어를 통해 TS를 직관적으로 다루길 원함.
- 기존 연구는 대부분 ***단일 모달리티(task-specific)*** 기반 → **자연어 기반 상호작용 부족**.

<br>

## (2) **Proposal**

- TSQA 정의: 사용자가 TS를 기반으로 다양한 질의를 자연어로 제시.

- 이를 위해 [Dataset] & [Model]을 제안

  1. [Dataset] **EngineMT-QA**: 센서 + 텍스트 기반 Multi-task QA 데이터셋
  2. [Model] **ITFormer**: LLM과 TS Encoder를 연결하는 **lightweight alignment 구조** 제안 
  

<br>

# **3. Related Works**

- **TS 분야**: forecasting, classification, anomaly detection 등 task-specific 연구가 중심.
- **LLM + Multi-modal QA**: 주로 vision-text 중심으로 발전 (e.g. VQA, VisualDialog).
- **TS + NLP**: Time-LLM, ChatTime 등 일부 시도는 존재하나, 텍스트를 보조정보로만 활용.
- **TSQA**는 **시계열 + 자연어 간 시맨틱 상호작용을 요구하는 새로운 패러다임**.

<br>

# **4. Methodology**

## **(1) Problem Definition**

- Input: 시계열 $$T$$, 자연어 질문 $$q$$
- Output: 자연어 답변 $$a$$
- Model: $$f : (T, q) → a$$ 를 학습

<br>

## **(2) ITFormer 구조**

![image-20250827140218077](/Users/seunghan96/Library/Application Support/typora-user-images/image-20250827140218077.png).

- **TPE (Time Token Position Encoding)**: 시간, 채널, 세그먼트 위치정보 부여
- **LIT (Learnable Instruct Token)**: task-specific semantic instruction
- **ITA (Instruct Time Attention)**: temporal & textual representation 동적 정렬
- **TAL (Time Token as Language)**: 시계열을 언어 토큰처럼 변환하여 LLM에 입력

<br>

## **(3) 학습 방법**

- Freeze: TS encoder, LLM
- Train (SFT): Alignment module (전체 파라미터 중 0.07%만 학습)
- Cross-entropy loss 사용 

<br>

# **5. Experiments**

## **(1) Dataset: EngineMT-QA**

- 기반: **NASA N-CMAPSS** 엔진 데이터

- QA 수: **110,000+**

- 총 4가지 태스크 포함:

  1. **Understanding** (open QA)
  2. **Perception** (fault classification)
  3. **Reasoning** (degradation 추론)
  4. **Decision-Making** (maintenance 판단)
  

<br>

## (2) **Evaluation Metrics**

- Open-ended: **BLEU, ROUGE-L**
- Classification: **Accuracy, F1-score**

<br>

## (3) **Baselines**

- Multimodal API: ChatGPT-4o, Gemini
- Vision-text: InstructBLIP, CoCa, MCAN-VQA
- TimeSeries-text: Time-LLM, AutoTime
- TS encoder: PatchTST (공통 사용)

<br>

## (4) **Results**

- ITFormer-7B이 모든 task에서 **최고 성능 달성**

  - Reasoning F1: 88.69 / Decision BLEU: 38.68
  - Vision-text, LLM API, 기존 TS-text 모두 능가
  
- Ablation 결과: TPE와 ITA가 가장 기여도가 큼

- **모듈 수 증가에 따라 성능 상승**

- **Efficiency 측면에서도 기존 방식 대비 inference 속도 향상** (Fig. 6 참조) 

<br>

## **6. Conclusion**

- **ITFormer**는 시계열과 자연어 사이의 semantic bridge 역할을 수행.
- **소수의 param update만으로도 강력한 QA 성능** 확보.
- **EngineMT-QA**는 향후 Time-Series QA 분야의 표준 데이터셋이 될 수 있는 잠재력.
