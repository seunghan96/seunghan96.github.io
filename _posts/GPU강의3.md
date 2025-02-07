---
title: Large Language Models; A Survey
categories: [MULT, LLM, NLP]
tags: []
excerpt: arxiv 2024
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Large Language Models: A Survey

<br>

### Contents

1. 3. 

<br>

# 1. 분산처리 기법 소개

## (1) 분산처리란?

- "하나"의 작업을 **"여러" 대의 GPU**로 나눠서 병렬로 처리
- **대규모 데이터** / 복잡한 계산을 주로 분산시킴
- 두 가지 종류
  - (1) **모델** 병렬화
  - (2) **데이터** 병렬화

<br>

![figure2](/assets/img/llm/img363.png)

<br>

## (2) LLM에서 분산처리의 필요성

1. **모델 size**
   - 수십~수천억개의 파라미터
   - 12B만되어도 single GPU에서는 많은 한계
   - 긴 context (cox) length를 위해서 필요
   - FFT 진행을 위해서는 필수적.
2. **성능 한계 극복**
   - 더 많은 자원 활용을 통한 성능 높이기
3. **시간 효율성**
   - Batch size를 더 크게 가져갈 수 있음
4. **자원 최적화**
   - 다양한 서비스 및 고도화된 서비스 deploy 가능

<br>

# 2. 분산기법의 종류

## (1) 개요

- a) Data Parallelism (DP)
- b) Model Parallelism (MP)
  - Tensor Parallelism (TP)
  - Pipeline Parallelism (PP)
- c) Distributed Data Parallelism (DDP)
- d) Fully Sharded Data Parallel (FSDP) $$\rightarrow$$ 설명 생략
- e) Zero Redundancy Optimizer (ZeRO) $$\rightarrow$$ 설명 생략

<br>

## (1) Data Parallelism

핵심: ***데이터를 분산***시켜서 학습

- 모델을 multi-GPU에 복제해야함
- **속도** 최적화 > 메모리 최적화
- (모델 사이즈가 커지는) 최근의 LLM 트렌드와는 거리감 O

![figure2](/assets/img/llm/img364.png)

<br>

## (2) Model Parallelism

핵심: ***모델을 분산***시켜서 학습

- 각 모델의 모듈을 각각의 GPU에 나눠서 학습
- 큰 모델 처리하는데에 유용
  - `transformers` 라이브러리에서 쉽게 사용 가능
- **메모리** 최적화 > 속도 최적화

![figure2](/assets/img/llm/img365.png)

<br>

### a) Tensor Parallelism

모델의 "각 텐서"를 여러 GPU에 분산

- ex) 행렬 연산

![figure2](/assets/img/llm/img367.png)

<br>

### b) Pipeline Parallelism

모델의 "여러 층"을 각 gpu에 나눔

- `DeepSpeed`: Pipeline Parallelism를 지원하는 라이브러리

![figure2](/assets/img/llm/img366.png)

<br>

## (3) DP vs. DDP

**Data Parallelism (DP)**

1. **개념:**

- 전체 모델을 **모든 GPU에 동일하게 복제**한 후, **미니배치를 여러 GPU에 나누어 처리**.
- 각 GPU에서 **독립적으로 forward & backward 연산을 수행**한 후, **모든 GPU의 그래디언트(gradient)를 중앙에서 모아서 평균을 낸 후 업데이트**.

2. **단점:**

- **중앙 서버(Parameter Server)가 필요**하며, 모든 GPU가 그래디언트를 전달하고 받아야 하므로 **통신 병목이 발생**할 수 있음.
- 보통 **싱글 노드(한 대의 서버)에서만 사용**됨.

3. **예시 (DP 적용 방식)**

- 배치 크기 128 → GPU 4개 → 각 GPU가 **128/4 = 32개 샘플**을 처리.
- 모든 GPU에서 **모델을 독립적으로 실행**한 후, 그래디언트를 중앙에서 모아 업데이트.

<br>

**Distributed Data Parallelism (DDP)**

1. **개념:**

- DP와 비슷하지만, **각 GPU가 직접 서로 통신(All-Reduce)하여 그래디언트를 공유**.
- **중앙 서버 없이** GPU들이 **분산된 방식으로 학습을 수행**.
- 싱글 노드뿐만 아니라 **멀티 노드(여러 서버) 환경에서도 확장 가능**.

2. **장점:**

- 중앙 서버 없이 각 GPU가 직접 통신하므로, **통신 병목이 줄어듦**.
- 멀티 노드(서버 여러 대)에서도 작동 가능하여, **대규모 학습에 적합**.

3. **예시 (DDP 적용 방식)**

- 배치 크기 128 → GPU 4개 → 각 GPU가 **32개 샘플**을 처리.
- 모든 GPU에서 **모델을 독립적으로 실행**한 후, **서로 그래디언트를 공유(All-Reduce)하여 평균을 내고 업데이트**.

<br>

# 3. `Axolotl` 라이브러리

Goal: ***LLM을 쉽게 Fine-tuning하기 위해***

( Hugging face와의 모델들과의 호환성 good )

- https://github.com/axolotl-ai-cloud/axolotl

<br>

### 주요 기능

1. **Model finetuning**
   - LoRA, QLoRA, GPTQ 등 
2. **Multi-GPU 지원**
   - DeepSpeed & FSDP
3. **Flash Attention, xformers, ROPE 등의 기술 통합**
4. **Dataset**
   - JSONL 같은 다양한 포멧 지원
