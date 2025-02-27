---
title: PPO in RLHF vs DPO
categories: [NLP, LLM, MULT, RL]
tags: []
excerpt: Proximal Policy Optimization, Direct Preference Optimization
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# 1. PPO (Proximal Policy Optimization) in RLHF

한 줄 요약: **사람의 피드백을 반영한 reward model**을 따로 학습한 후, 이를 이용해 **PPO로 정책 최적화**!

<br>

## Loss function

**기존 정책과의 차이를 제한**하면서 **보상을 최대화**하는 손실 함수를 사용

$$L(\theta) = \mathbb{E} \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t) \right]$$.

- $$r_t(\theta) = \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$$: 새로운 정책과 기존 정책의 확률 비율(importance sampling ratio)
- $$A_t$$ :특정 행동이 얼마나 좋은지를 나타내는 **Advantage 함수**
  - $$A_t$$는 **reward model이 예측한 보상**을 기반으로 계산
- $$\epsilon$$: 정책이 급격하게 변하는 것을 방지

<br>

요약: 

- PPO 자체가 reward model을 포함 X
- But RLHF에서는 사람이 부여한 선호 데이터를 학습한 reward model 이용해 PPO를 최적화하는 방식으로 사용

<br>

# **2. DPO (Direct Preference Optimization)**

한 줄 요약: **reward model 없이** 직접 선호 데이터를 기반으로 최적화

<br>

## Loss function

DPO는 사람이 **선호한 답변**과 **그렇지 않은 답변**을 비교하면서, 좋은 답변을 더 높은 확률로 생성하도록 학습

$$L(\theta) = \mathbb{E}_{(x^+, x^-) \sim D} \left[ -\log \sigma(\pi_{\theta}(x^+) - \pi_{\theta}(x^-)) \right]$$.

- $$\pi_{\theta}(x)$$: 모델이 답변 $$x$$를 생성할 확률의 로그 값 (로그 확률 $$\log P_{\theta}(x)$$)

요약

- 이 수식은 **좋은 답변 $$x^+$$의 확률이 나쁜 답변 $$x^-$$보다 높아지도록 모델을 학습**
- PPO와 달리 **reward model을 따로 학습하지 않고도 직접 선호 데이터만으로 최적화가 가능**해.

<br>

# 3. PPO vs. DPO 비교

| 항목          | PPO                               | DPO                              |
| ------------- | --------------------------------- | -------------------------------- |
| reward model  | 따로 학습한 reward model을 활용   | reward model 없이 직접 최적화    |
| 정책 업데이트 | reward model을 기반으로 강화 학습 | 선호 데이터를 기반으로 지도 학습 |
| 손실 함수     | (보상 기반) Advantage 최적화      | 선호 비교를 직접 최적화          |
| 계산 비용     | 상대적으로 큼 (샘플 효율이 낮음)  | 비교적 작음 (샘플 효율이 높음)   |

<br>

## Summary

1. **PPO (in RLHF)**
   - 사람이 특정 데이터셋에 대해 **선호/비선호 혹은 선호 정도를 레이블링**
   - 이 데이터를 활용해 **보상 모델을 학습**
   - 보상 모델을 사용하여 PPO를 통해 **정책을 최적화**
2. **DPO**
   - 사람이 특정 데이터셋에 대해 **선호/비선호를 레이블링**
   - 보상 모델을 따로 만들지 않고, **선호 데이터로 직접 최적화**

즉, **PPO는 보상 모델을 거쳐** 정책을 최적화하고, **DPO는 직접 선호 데이터를 활용해** 정책을 최적화!
