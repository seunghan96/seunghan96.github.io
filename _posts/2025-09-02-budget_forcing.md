---
title: Test-time Scaling (TTS) - Budget Forcing
categories: [LLM, MULT]
tags: []
excerpt: TTS, Budget Forcing
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Test-time Scaling (TTS) - Budget Forcing

## Contents

1. Budget Forcing (BF)
2. Token-Conditional Control (TCC)
3. Step-Conditional Control (SCC)
4. Class-Conditional Control (CCC)
5. Rejection Sampling (RS)

<br>

# 1. **Budget Forcing (BF)**

(**Reference**: *s1: Simple test-time scaling* (2025) – https://arxiv.org/abs/2501.19393)

- Model이 지정된 **추론 token 예산 (budget)** 을 채우도록 **"강제"**
  - Loop에 빠지면 일정 시점에서 **강제 마무리**

- 간단 + 제어력 good + 성능 안정적

<br>

# 2. **Token-Conditional Control (TCC)**

- Prompt에 ***“몇 개 token 이상/이하로 생각하라”***를 조건으로 줌

- 생성 길이를 **"간접적"**으로 유도

  $$\rightarrow$$ **정확한 제어 어려움**!

- 추론 길이에 따라 **성능이 불안정**

<br>

# 3. **Step-Conditional Control (SCC)**

- Prompt에 ***“몇 단계 (step)로 풀라”***는 조건을 명시

  (= Reasoning "단계" 수를 제약)

  $$\rightarrow$$  **실제 token 수와 불일치** 가능 (제어력이 떨어짐)

<br>

# 4. **Class-Conditional Control (CCC)**

- 입력 문제를 난이도/카테고리(class)별로 나눠 **각 class별 다른 예산/깊이 요구**
  - 문제 특성에 맞는 맞춤 제어 가능

- Labeling 필요 & generalization 부족

<br>

# 5. **Rejection Sampling (RS)**

- **여러 후보**를 생성 후 조건(길이/정답 형태)에 맞지 않으면 **버리고 반복**

  $$\rightarrow$$ 조건 충족할 때까지 반복하므로 **비용↑**

- 길이가 길수록 오히려 오류 가능성↑ 

<br>

# Comparison

| **방법** | **제어 단위**             | **장점**                     | **단점**                     |
| -------- | ------------------------- | ---------------------------- | ---------------------------- |
| **BF**   | token 예산(budget) 강제   | 단순, 제어력 100%, 성능 우수 | 예산 이상 확장 불가          |
| **TCC**  | token 수 조건 (Prompt)    | 구현 쉬움                    | 실제 길이 제어 불안정        |
| **SCC**  | 추론 단계(step) 조건      | 직관적, 사람 이해 쉬움       | token 수와 불일치, 제어 약함 |
| **CCC**  | 문제 클래스별 조건        | 문제 특성별 맞춤 가능        | 라벨 필요, 일반성 부족       |
| **RS**   | 조건 충족까지 반복 샘플링 | 정답률 개선 가능             | 비용↑, inverse scaling 발생  |
