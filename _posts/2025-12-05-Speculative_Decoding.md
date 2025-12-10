---
title: Speculative Decoding
categories: [LLM]
tags: []
excerpt: 

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Speculative Decoding

## 1. 간단 요약

for **큰 LLM**의 속도를 빠르게 하기 위해!

- Step 1) **작은 LLM**이 미리 후보 token을 "여러" 개 예측하고

- Step 2) **큰 LLM**이 이를 빠르게 검증하는 방식 (채택 or 기각)

<br>

***Speculative*** = “추측에 기반한, 미리 예상해서 하는”

<br>

## 2. 주요 개념

두 종류의 model

- 작은 LLM = **Draft model (초안 생성기)**
- 큰 LLM = **Target model (정확한 최종 model)**

<br>

작은 LLM이 **“미리 뱉은 여러 token 후보”** 중

**"큰 LLM이 통과시키는 것만 채택"**해서 한 번에 여러 token을 생성함

→ ***큰 LLM이 매 token을 하나씩 생성할 필요가 없어져 속도가 크게 증가***

<br>

## 3. 필요성

LLM에서 가장 **느린 작업**?

```
1 token 생성 → 다시 model forward pass → 또 1 token 생성 …
```

이 **“token-by-token” 방식**이 병목!

<br>

Speculative decoding

- 작은/빠른 LLM이 **여러 token (batch)**을 미리 예측
- 큰 LLM은 **검증만**! 
  - 맞으면 그대로 채택
  - 틀리면 되돌아가서 다시 생성

<br>

## 4. 동작 방식 (Detail)

- Step 1) **작은 LLM(Draft)**이 token을 k개 미리 예측

```
Draft: "The weather today is very"
→ ["very", "nice", "and", "quite", "warm"]
```

<br>

- Step 2) **큰 LLM(Target)**은 이 후보들이 “가능성 있는지”만 빠르게 체크

  - feat. Probability ratio test

  - 큰 LLM이 ok라고 판단하면

    → 한 번에 여러 token을 “확정”

    → 속도 매우 빠름

  - 큰 LLM이 reject하면

    → 그 지점에서부터 큰 LLM이 직접 생성 (fallback)

<br>

## 5. 장점

(1) 속도 대폭 향상

- 초안이 여러 개 → 검증은 빠름 → 전체 latency 감소
- 실전에서 **2~4배 속도 향상** 자주 관측됨

<br>

(2) 정확도 영향 없음

- “큰 LLM이 최종 검증”하므로

  → 품질(target model과 동일) 보장

<br>

(3) 작은 LLM 재사용 가능

- 작은 LLM은 Distillation 같은 작업 필요 없음
- 별도 추가 학습 없이 그냥 사용 가능

<br>

## 6. 단점

- 후보를 많이 생성하면 검증 비용 증가

  → 최적 k 선택이 필요

- Draft가 너무 나쁘면 rejection률이 높아져 속도 향상 줄어듬!

- 두 model의 tokenization과 vocab이 동일해야 함