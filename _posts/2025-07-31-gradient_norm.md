---
title: Gradient Norm
categories: [LLM, MULT]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Gradient Norm

## Contents

1. 개요
2. (학습 중) 체크 이유
3. 높을 때 / 낮을 때 
4. 적당한 값?
5. Summary

<br>

# 1. 개요

학습 시, 각 parameter에 대한 **loss의 기울기(gradient)**가 계산됨

모델 전체 parameter의 gradient를 하나의 벡터처럼 생각했을 때...

$$\rightarrow$$  그 벡터의 **크기(norm)**를 gradient norm이라고 함!

<br>

(주로 L2 norm 사용)

- $$\mid \mid g\mid \mid _2 = \sqrt{\sum_i g_i^2}$$.

<br>

# 2. (학습 중) 체크 이유

**gradient norm**

= “현재 학습 단계에서 ***model parameter가 얼마나 크게 update*** 될지”를 보여주는 지표

<br>

(Optimizer가 step을 진행할 때)

$$\Delta \theta \approx - \eta \, g$$.

- $$\eta$$ learning rate
- $$g$$: gradient

$$\rightarrow$$ 즉, $$\mid \mid g\mid \mid$$ 이 크면 parameter가 크게 바뀌고, 작으면 거의 안 바뀜

<br>

# 3. 높을 때 / 낮을 때 

## (1) 높은 경우

현상: **gradient explosion**

- **parameter update가 과도** → 학습 불안정, loss NaN, 발산(diverge)

<br>

해결

- **gradient clipping** 
  - e.g., $$\mid \mid g\mid \mid _2 > \tau$$이면 $$\tau$$로 clipping
- learning rate 줄이기
- 안정적 초기화/정규화(RMSNorm, LayerNorm 등)

<br>

## (2) 낮은 경우

현상: **gradient vanishing**

- **update가 거의 없음** → 학습이 느리거나 멈춤
- deep NN/ sigmoid 계열에서 자주 발생

<br>

해결

- ReLU, GELU 같은 활성화 함수 사용
- Residual connection
- 적절한 weight 초기화
- Normalization 기법

<br>

# 4. 적당한 값?

당연히 **절대적인 “좋은 값”은 없음**

- model size, parameter 수, loss scale, batch size에 따라 norm 크기가 dependent

중요한 건 **추세와 안정성**입니다:

- 학습 내내 norm이 일정한 범위에서 안정적으로 유지 → OK
- 갑자기 폭발(수천 이상) → 발산 위험
- 계속 0 근처에 머무름 → 학습 정체

<br>

# 5. Summary

**gradient norm** = 전체 기울기의 크기 (update 세기 지표)

- **높으면**: 발산 위험 → clipping 필요
- **낮으면**: 학습 정체 → 학습률/모델 구조 개선 필요

**적당한 값**은 절대적 기준보다 **안정적 범위 유지**가 중요

