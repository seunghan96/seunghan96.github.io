---
title: (Quantization) (1) Floating Point
categories: [LLM, MULT]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Contents

1. 부동소수점 표현 공식 (IEEE 754)
2. Example) FP32
3. Fraction 설명

<br>

# 1. 부동소수점 표현 공식 (IEEE 754)

$$V = (-1)^{\text{sign}} \times (1.\text{fraction})_2 \times 2^{(\text{exponent} - \text{bias})}$$.

$$V = (-1)^s \times (1 + \sum_{i=1}^n f_i 2^{-i}) \times 2^{(e-b)}$$

- (s) **sign**: 부호 bit 
  - 0이면 양수
  - 1이면 음수

- (e) **exponent**: 지수부 
  - bias를 적용해야 (빼줘야)!!
  - **범위를 결정**하는 핵심 요소

- (c) **bias**: 지수의 중앙값
  - (FP32) bias = 127 (범위: 1-127 ~ 254-127)
  - (FP16) bias = 15
  - (bfloat16) bias = 127
- (d) **fraction (mantissa)**: 유효 숫자부
  - 2진수 소수로 해석. 
  - 맨 앞 "1"은 **정규화**에서 항상 implicit하게 존재


<br>

| **형식** | **구성 (s/e/f)** | **공식**                                 | **값 범위**    | **예시 (3.14)** | **바이트** |
| -------- | ---------------- | ---------------------------------------- | -------------- | --------------- | ---------- |
| FP32     | 1 / 8 / 23       | $$(-1)^s (1+\sum f_i 2^{-i}) 2^{e-127}$$ | $$±3.4×10^38$$ | 3.1415926       | 4 B        |
| FP16     | 1 / 5 / 10       | $$(-1)^s (1+\sum f_i 2^{-i}) 2^{e-15}$$  | $$±6.55×10^4$$ | 3.1406          | 2 B        |
| bfloat16 | 1 / 8 / 7        | $$(-1)^s (1+\sum f_i 2^{-i}) 2^{e-127}$$ | $$±3.4×10^38$$ | 3.140625        | 2 B        |
| INT8     | 8비트 정수       | $$-b_7·2^7 + \sum b_i 2^i$$              | $$-128~127$$   | 3               | 1 B        |
| INT4     | 4비트 정수       | $$-b_3·2^3 + \sum b_i 2^i$$              | $$-8~7$$       | 3               | 0.5 B      |

<br>

# 2. Example) FP32

## (1) s/e/b/f

- 부호 $$s$$: 1 비트

- 지수 $$e$$: 8 비트 → 0~255 표현 가능
  - (참고) "지수" 그대로 사용하는 것이 아니라, "bias"를 빼줘야 함.
  
    $$\rightarrow$$ 즉, 실제 지수는 $$E = e - 127$$
  
- bias $$b$$ = 127

- fraction $$f$$: 23 비트

<br>

## (2) 지수 범위

FP32는 지수를 8비트로 표현하므로,

$$\rightarrow$$ 0 (=0000000) ~ 255 (=11111111)을 표현할 수 있음

<br>

### 최소: $$e=1$$

$$E = e(1)-127 = -126$$

- 왜 $$e=0$$이 아닌지? 
  - 이 경우는 **subnormal(비정규화 수)** 또는 **0**을 표현하는 데 사용

<br>

### 최대: $$e=254$$

$$E = e(254)-127 = +127$$

- 왜 $$e=255$$이 아닌지?
  - 이 경우는 **특수 값**에 사용
    - fraction = 0 → $$+\infty, -\infty$$
    - fraction ≠ 0 → NaN (Not a Number)

<br>

### FP32: $$2^{-126} \leq 2^E \leq 2^{127}$$

<br>

## (3) 유효숫자 (fraction) 범위

fraction은 항상 **1.xxx** 형태 

- 최소: 1.0 
- 최대: 거의 2.0 
  - $$1.{111…1}_2$$.
  - $$1 + 0.{111…1}_2$$.
  - $$1+(2^{-1}+2^{-2}+...+2^{-32})$$.
  - $$2 - 2^{-23}$$.


<br>

따라서, 유효숫자(mantissa)는, $$1 \leq (1+\sum f_i 2^{-i}) < 2$$.

<br>

## (4) 최댓값

$$V = (-1)^s \times (1 + \sum_{i=1}^n f_i 2^{-i}) \times 2^{(e-b)}$$.

- a) $$(-1)^s=(-1)^0 = 1$$
- b) $$(1 + \sum_{i=1}^n f_i 2^{-i}) \approx 2$$
- c) $$2^{(e-b)} = 2^{254-127} = 2^{127}$$

<br>

$$V_{max} \approx 2 \times 2^{127} = 2^{128}$$.

$$\rightarrow$$ 10진수로 변환 시 $$2^{128} \approx 3.4 \times 10^{38}$$

<br>

## (5) 최솟값 (양수)

$$V = (-1)^s \times (1 + \sum_{i=1}^n f_i 2^{-i}) \times 2^{(e-b)}$$.

- a) $$(-1)^s=(-1)^0 = 1$$
- b) $$(1 + \sum_{i=1}^n f_i 2^{-i}) \approx 1$$
- c) $$2^{(e-b)} = 2^{1-127} = 2^{-126}$$

<br>

$$V_{min} \approx 1 \times 2^{-126}$$.

$$\rightarrow$$ 10진수로 변환 시 $$1.18 \times 10^{-38}$$.

<br>

# 3. Fraction 설명

왜 $$1 \leq (1 + \Sigma f_i 2^{-i}) < 2$$인지 ?

<br>

## (1) Fraction (Mantissa)란?

$$V = (-1)^s \times (1 + \text{fraction}) \times 2^{(e - bias)}$$,

$$\rightarrow$$ 여기서 **fraction**은 $$f_1 f_2 f_3 … f_n$$ (0 또는 1 bit들의 나열)이고, 

$$\rightarrow$$ 실제 값은 $$\text{fraction} = \sum_{i=1}^n f_i \cdot 2^{-i}$$

<br>

즉, 이진수 소수점으로 “0.xxx…” 부분을 표현하는 거!

<br>

## (2) 왜 항상 1이 붙나? (정규화 normalized number)

IEEE 754의 정규화 규칙

- 가장 앞자리(leading bit)는 항상 1이 되도록 저장.
- 그래서 저장할 때는 이 1을 따로 안 적고, 계산할 때만 **implicitly 1.**을 붙입니다.

$$\rightarrow$$ 즉, 실제 유효숫자 = $$1 + \text{fraction}.$$

<br>

## (3) 범위 계산

fraction이 전부 $$0$$일 때:

- $$1 + \sum f_i 2^{-i} = 1.0$$.

fraction이 전부 1일 때 (예: 23비트 전부 1 in FP32):

- $$1 + \left(2^{-1} + 2^{-2} + … + 2^{-23}\right) = 1 + (1 - 2^{-23}) = 2 - 2^{-23}$$

<br>

즉, **최소 1.0 ≤ mantissa < 2.0** 이 되는 거예요.

<br>

## (4) Example (FP32)

- fraction = 000...0 → mantissa = 1.0
- fraction = 100...0 → mantissa = 1 + 0.5 = 1.5
- fraction = 111...1 → mantissa = 1.111…_2 = 1.99999988…

