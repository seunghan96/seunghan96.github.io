---
title: Chain-of-Thought Reasoning without Prompting
categories: [LLM, MULT]
tags: []
excerpt: NeurIPS 2024
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Chain-of-Thought Reasoning without Prompting (NeurIPS 2024)

Wang & Zhou, Google DeepMind, arXiv 2402.10200v2, 2024

<br>

***“Prompt 없이도 LLM이 본래 갖고 있는 추론 능력을 끌어낼 수 있다”***

***“LLM은 Prompt가 없어도 CoT 추론 경로를 본래 품고 있으며, 단지 greedy decoding에 가려져 있었음!. Decoding을 바꾸면 이 추론이 드러난다.”***

<br>

# 1. Background

- 지금까지 LLM의 **추론(Reasoning)** 성능은 대부분 **CoT prompting**을 통해 드러남

  - e.g., *“단계를 나눠서 풀어라”* 같은 지시문을 Prompt에 넣기

- 하지만 이는 **Prompt engineering**에 의존

  $$\rightarrow$$ LLM의 “본질적 추론 능력”을 정확히 평가하기 어려움

<br>

# 2. Key Idea

**질문 그대로**(QA 형식) + Decoding만 바꿔봄.

- 보통은 **Greedy decoding (top-1 계속 선택)**을 하는데,
- 여기서 **"top-k 대안 경로"**를 살펴보면, **자연스럽게 CoT 추론 경로가 숨어 있음**!!

<br>

즉, **추론 경로**를 내부에 이미 가지고 있지만, greedy decoding 때문에 잘 드러나지 않았던 것일 뿐!!!

$$\rightarrow$$ CoT-decoding

<br>

## Contributions

1. **Prompt 없이도 LLM은 CoT 추론 경로를 본래 가지고 있음**을 확인
2. **Decoding 전략만 바꿔서** LLM의 내재적 추론 능력을 평가
3. **Confidence 기반 CoT-decoding** 제안 → Prompt 없이도 Self-consistency 같은 효과 달성.
4. 모델 크기, 튜닝 여부와 관계없이 일관된 향상

<br>



<br>

# 3. CoT-Decoding

1. **Top-k branching**

   - 첫 번째 Decoding 단계에서 **"top-k 후보"**를 여러 개 뽑고 각각 경로를 탐색
   - Greedy path는 종종 오답을 내지만, **"다른 경로에는 CoT 추론이 존재"**함을 확인

2. **Confidence 기반 선택**

   - **CoT가 "있는" 경로**일수록 **"최종 답의 token 확률 차이(Δ)"**가 커서 **더 확신(confidence)**을 보임
   
   $$\rightarrow$$ Idea) 이를 활용해 CoT-path를 선택하자!!
   
   

<br>

# 4. Details

CoT-decoding의 수식

<br>

## (1) 기본 세팅

- 질문 입력: $$x$$
- 답변 출력: $$y = (y_1, y_2, \ldots, y_T)$$
- LLM의 조건부 확률: $$P(y\mid x) = \prod_{t=1}^T P(y_t \mid x, y_{<t})$$

<br>

## (2) Greedy decoding

각 단계에서

- $$y_t = \arg\max_{v} P(v \mid x, y_{<t})$$.

을 선택

→ 항상 최빈 token만 이어져서, **“즉답(short answer)”**으로 끝날 가능성이 큼

<br>

## (3) CoT-Decoding: Branching

Step 1) CoT-decoding의 **첫 단계 t=1**:

- Top-k 후보를 여러 개 뽑는다!

- $$\mathcal{Y}_1 = \{ y_1^{(1)}, y_1^{(2)}, \dots, y_1^{(k)} \}, \quad y_1^{(i)} \sim \text{Top-k}\big(P(\cdot \mid x)\big)$$.

<br>

Step 2) 각 $$i$$ (for $$i$$ in [1,,$$k$$]) 에 대해  $$y_1^{(i)}$$를 시작점으로 **독립적인 경로(trajectory)**를 샘플링

- $$y^{(i)} = (y_1^{(i)}, y_2^{(i)}, \dots, y_{T_i}^{(i)})$$.

<br>

Step 3) 각 $$i$$ (경로)에 대해 Confidence 계산

- 아래 참조

<br>

## (4) CoT-Decoding: Confidence

각 후보 답변 $$y^{(i)}$$의 마지막 답 token(예: 최종 숫자 정답)의 확률을 확인

- 논문에서는 **confidence**를 “정답 token과 그 다음 token의 log 확률 차이”로 정의

  (=  **정답 token의 확률과 2위 후보 token의 확률 차이** )

<br>

$$\Delta_{k,\text{answer}} = \frac{1}{\mid \text{answer}\mid } \sum_{x_t \in \text{answer}} \big( P(x_t^1 \mid x_{<t}) - P(x_t^2 \mid x_{<t}) \big)$$

- $$x_t^1$$ = 그 시점에서 가장 확률 높은 token,
- $$x_t^2$$ = 두 번째로 높은 token.

$$\rightarrow$$ 즉, “1등 token이 2등보다 얼마나 더 확률이 높은가”를 보는 것.

<br>

### 직관

- **확신이 크다** = 답 token이 다른 후보보다 압도적으로 확률이 높음.
- **확신이 작다** = 답 token과 다른 후보가 비슷한 확률 → 불확실

<br>

### Example: 산수 문제 “123+456=?”

[후보 1] **Greedy decoding** 경로: "579"

- 마지막 숫자 "9"의 확률 = 0.36
- 두 번째 후보 "8"의 확률 = 0.34

차이 = 0.02 → **불확실**

$$\rightarrow$$ 이 경로는 reasoning 없이 그냥 답 찍기.

<br>

[후보 2] **CoT decoding 경로**: "Let's think step by step. 123+456=579. The answer is 579."

- 마지막 "9"의 확률 = 0.82
- 두 번째 후보 "8"의 확률 = 0.07

차이 = 0.75 → **매우 확실**

$$\rightarrow$$ 이 경로는 reasoning을 거쳤기 때문에 답에 훨씬 자신감 있음.

<br>

## (5) Summary

- Greedy: $$y^* = \arg\max P(y_1tokenx)$$ 경로 → “즉답”으로 빠지기 쉬움.

- CoT-decoding: 첫 token 분기를 열고 → 각 경로를 탐색

   → **confidence 기반 선택**으로 자연스럽게 Chain-of-Thought 경로를 찾음.