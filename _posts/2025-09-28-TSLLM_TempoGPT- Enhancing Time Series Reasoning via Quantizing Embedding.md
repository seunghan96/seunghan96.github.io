---
title: TempoGPT: Enhancing Time Series Reasoning via Quantizing Embedding
categories: [LLM, MULT, TS]
tags: []
excerpt: arxiv 2025

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# TempoGPT: Enhancing Time Series Reasoning via Quantizing Embedding (arxiv 2025)

- 인용 수: (2025-08-27 기준) 0회
- 링크: https://arxiv.org/pdf/2501.07335



# **1. Abstract**

기존 TLM (Time-series Language Models)

- (O) Trend 분석에는 성능이 좋지만
- (X) **복잡한 reasoning task**에선 성능 저하. 

<br>

원인:

1. 학습 데이터에 ***reasoning 과정이 포함되지 않음***
2. TS는 정교한 tokenization이 없어, Text와 표현 방식이 불일치.

<br>

제안: **TempoGPT**

- White-box 시스템 기반의 **Multi-modal TS reasoning dataset** 구성.
- **Temporal embedding을 정량화(quantization)**하여 discrete token으로 변환.
- Temporal + Text token을 **공유 embedding layer**로 처리.

<br>

Experiment

- Reasoning task에서 **state-of-the-art 성능** 및 논리적 일관성 확보.

- 코드: https://github.com/zhanghaochuan20/TempoGPT

<br>

# **2. Introduction**

## (1) **Motivation**

- TLM은 vision, audio 등에서 성공했지만 ***TS reasoning에는 제한적 성능***

  (특히, **물리 시스템과 변수 간 관계 추론**은 어려움)

- 기존 Multi-modal TS는 대부분 **단순 label만 존재** 

  $$\rightarrow$$ ***Reasoning 학습 불가***

<br>

## (2) **Proposal**

- **White-box system 기반 데이터 생성** (물리 시뮬레이터 활용).
- **Temporal quantization** 기법을 통해 TS 정보를 discrete token으로 정제.
- Token 간 표현 정합성을 맞춰 LLM의 reasoning 능력 활용 극대화.

<br>

![image-20250827135159151](/Users/seunghan96/Library/Application Support/typora-user-images/image-20250827135159151.png)

.

# **3. Related Works**

## **(1) Time Series Language Models**

- 초기) LLM을 forecasting/classification 등 **단순 task**에 사용.
- 중기) 이후 Prompt 기반 혹은 Alignment 기반 방법 등장.
- 최근) Trend, seasonality 등 **기초 TS 특성 추출에는 성공**

$$\rightarrow$$ 그러나 **복잡 reasoning에선 부진** !!

<br>

## **(2) Multi-modal Alignment**

- Vision 도메인 기반 alignment(BLIP-2, CLIP 등)를 차용한 시도.
- TS은 **tokenization이 없어 표현 일관성 부족** → LLM 활용에 한계.
- 일부 연구에서 quantization 도입 시도 있으나, 대부분 단순 예측 용도에 그침.

<br>

# **4. Methodology**

## **(1) Multi-modal Data Construction**

- **White-box 전기 회로 시뮬레이터**를 구성하여 6개 TS 변수를 생성:
  - 두 개의 AC 전원, 세 개의 저항, 전류 등.
  
- Rule-based + Human-in-the-loop 방식으로 label 생성.

<br>

**총 5가지 reasoning task**

1. Trend Analysis
2. Trend Forecast
3. Fault Judgement
4. Fault Diagnosis
5. Fault Analysis

![image-20250827135240510](/Users/seunghan96/Library/Application Support/typora-user-images/image-20250827135240510.png)

<br>

- Pretrain은 anomaly 정보와 temporal 정보 중심, Finetune은 CoT 방식 질문 포함.

<br>

## (2) TempoGPT 구조

![image-20250827135311773](/Users/seunghan96/Library/Application Support/typora-user-images/image-20250827135311773.png)

- **Quantization Encoder**: 
  - VQ-VAE 기반 patch-wise encoding 
  - Codebook으로 정수 token 변환.

- **Shared Embedding Layer**: 
  - 기존 Text vocab에 temporal token 확장.

- **LLM**: 
  - GPT-2, LLaMA, Phi-2 등 다양한 backbone 사용 가능.

- Procedure
  - Step 1) Pre-train (embedding W만 학습) 
  - Step 2) Fine-tune (LoRA or Full tuning).


<br>

# **5. Experiments**

## **(1) Dataset**

- 전기 시뮬레이터 기반 생성 데이터셋.

- 총 5가지 reasoning task에서 평가:

  - Trend 관련 (2개)
  - Reasoning 관련 (3개)
  

<br>

## (2) **Task**

- 자연어 QA 형식으로 주어지는 TS 해석 문제.

<br>

## (3) **Baseline**

- **Text Prompt 기반**: GPT-3.5, GPT-4 (TS을 Text로 변환)

- **Continuous Embedding 기반**:
  - GPT4MTS (Linear)
  - Time-LLM (Attention)
  - ChatTS (MLP)
  

<br>

## (4) **Evaluation Metric**

- **CA**: Conclusion Accuracy
- **LRA**: Logical Reasoning Accuracy
- **DR**: Deception Rate (결론은 맞지만 reasoning이 틀림)

<br>

## (5) **Resource**

- GPU: 4 × V100
- GPT-2는 full fine-tuning, 나머지는 LoRA 사용
- Pretrain: 6.6K step / Finetune: 7.2K step 

<br>

# **6. Conclusion**

TLM의 TS reasoning 한계를 해결하기 위해...

- **White-box 기반 데이터셋 생성**
- **Temporal quantization 기반 token alignment**

결론

- TempoGPT는 **논리적으로 일관된 reasoning**, **낮은 DR**, **높은 LRA**를 달성.