---
title: Prompt Tuning
categories: [LLM, NLP]
tags: []
excerpt: PEFT, Prompt Tuning
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Prompt Tuning

참고: https://blog.harampark.com/blog/llm-prompt-tuning/

<br>

### Contents

1. PEFT
2. Prompt Tuning
3. Prefix tuning vs. P tuning vs. Prompt Tuning

<br>

# 1. PEFT (Parameter-Efficient Fine-Tuning)

- 모델의 **"일부"**의 파라미터만을 효과적으로 FT함
- ex) `LoRA`, `Prefix Tuning`, `P-Tuning`, `Prompt Tuning`

<br>

# 2. Prompt Tuning

( [The Power of Scale for Parameter-Efficient Prompt Tuning](https://doi.org/10.48550/arxiv.2104.08691) )

<br>

## (1) Process: 간소화한 형태의 FT

- Downstream task를 수행하기 위해 모델의 파라미터를 고정
- "개별 Task"마다 prompt의 파라미터를 업데이트 ( = `Soft Prompt` )
  - Task별로 input 값에 prompt를 붙여서 학습

<br>

## (2) Prompt Tuning vs. Model Tuning

- Model Tuning: 모델의 "모든" 파라미터를 튜닝
- Prompt Tuning: 모델의 파라미터는 고정 +. 태스크 A와 B, C에 튜닝이 가능한 토큰을 붙인다.

<br>

## (3) Hard Prompt vs. Soft Prompt

Hard prompt

- input의 특성에 따라 input 값이 미리 정의된 텍스트 템플릿으로 바뀐다
- 이산적인(discrete) 값

Soft prompt

- input 앞에 튜닝이 가능한 임베딩 조각(tunable piece of embedding)이 붙게 된다
- 연속적인(continuous) 값

<br>

## (4) Prompt Tuning의 고려사항

Q1) Prompt의 초깃값

- (1) Random initialization
- (2) 개별 prompt 토큰을 모델의 vocabulary에서 추출한 임베딩으로 초기화
- (3) prompt를 출력 클래스를 나열하는 임베딩으로 초기화

<br>

Q2) Prompt의 길이를 얼마로 할 것인가?

Prompt의 길이를 위한 파라미터의 비용 = EP

- E: prompt 토큰 dimension
- P: prompt 토큰 길이

<br>

# 3. Prefix tuning vs. P tuning vs. Prompt tuning

## (1) Prefix tuning

Prefix tuning = 모든 트랜스포머 레이어에 접두어 시퀀스를 붙여 학습

Prefix Tuning vs. Prompt Tuning

- Prefix Tuning: 트랜스포머의 모든 계층에 접두어
- Prompt Tuning: input 앞에 접두어 => 더 적은 파라미터로 FT 가능
- ex) `BART`: 
  - Prefix tuning: 인코더와 디코더 네트워크에 모두 접두사
  - Prompt Tuning: 인코더의 프롬프트에만 접두어를

<br>

## (2) P-tuning

P-tuning = 인간이 디자인한 패턴을 사용

- 학습이 가능한 연속적인 prompt를 input 전체에 삽입

<br>

Prefix Tuning vs. Prompt Tuning

- P-tuning: 
  - a) 입력 전체에 연속적인 prompt를 삽입
  - b) 모델 파라미터도 같이 업데이트
- Prompt Tuning: 
  - a) 접두어로 붙이기만함
  - b) 모델 파라미터 고정
