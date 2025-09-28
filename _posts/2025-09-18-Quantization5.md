---
title: (Quantization) (5) Symmetric vs. Asymmetric quantization
categories: [LLM, MULT]
tags: []
excerpt: Q-LoRA (Quantized Low-Rank Adaptation)

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Contents

1. Symmetric vs. Asymmetric
2. Symmetric Quantization
3. Asymmetric Quantization
4. Summary
5. Example

<br>

# 1. Symmetric vs. Asymmetric

핵심: **“zero-point를 어떻게 다루느냐”** 차이

- "scaling factor": 차이 X
- "zero-point": 차이 O

  $$\rightarrow$$ zero-point 개념이 **붙느냐 vs. 안 붙느냐**로 구분!

<br>

[복습]

- **양자화 (quantize)**
  - $$q = \text{round}\left(\frac{x}{\text{scale}} + \text{zero\_point}\right)$$.

- **복원 (dequantize)**
  - $$\hat{x} = (q - \text{zero\_point}) \times \text{scale}$$.

<br>

# 2. Symmetric Quantization

**아이디어**: 실수 값 분포를 0을 중심으로 대칭적 (symmetric)으로 정수 범위에 매핑.

- 즉, $$[-max, +max]$$ → $$[-Q_{max}, +Q_{max}].$$
  - 이때 **zero-point = 0** (항상 원점이 정수 0에 매핑됨)
- 참고: 여기서 max는 "절대값 기준"
  - 즉, $$max$$ = **max(|min|, |max|)**

<br>

**scaling factor**:

- $$\text{scale} = \frac{\max(\mid x \mid)}{Q_{max}}$$.
- e.g., INT8: $$Q_{max} = 127$$

<br>

**장/단점**

- 장점: 구현이 단순, 곱셈만 필요.
- 단점: 값 분포가 0에 치우쳐 있으면 범위를 낭비

<br>

# 3. Asymmetric Quantization

**아이디어**: 값 분포가 0을 중심으로 대칭적이지 않을 때, **범위를 [min, max] 전체로 매핑**.

- 즉, $$[min, max] → [0, Q_{max}]$$

- **zero-point ≠ 0** → “offset”이 들어감


<br>

**scaling factor**:

- $$\text{scale} = \frac{\max - \min}{Q_{max} - Q_{min}}$$.

- **zero-point**:
  - $$\text{zero\_point} = -\text{round}\left(\frac{\min}{\text{scale}}\right)$$.

<br>

**장/단점**

- 장점: 분포에 더 잘 맞춰서 표현 손실 적음.
- 단점: 연산 시 (q - zero_point) 형태가 들어가서 계산이 조금 복잡.

<br>

# 4. Summary

**Scaling factor 계산 방식**은 케이스마다 조금 달라지지만,

- symmetric: **min/max 절대값** 기반
- asymmetric: **전체 min~max** 범위 기반

<br>

scaling factor 자체는 둘 다 “실수 범위 ÷ 정수 범위”라는 뼈대는 같음.

- **차이는 zero-point를 두느냐(= asymmetric)**, 그냥 0으로 고정하느냐(= symmetric).

<br>

| **구분**   | **Symmetric**  | **Asymmetric**   |
| ---------- | -------------- | ---------------- |
| 매핑 범위  | [-max, +max]   | [min, max]       |
| zero-point | 항상 0         | 0이 아닐 수 있음 |
| 장점       | 단순, 빠름     | 분포에 잘 맞음   |
| 단점       | 범위 낭비 가능 | 연산 복잡도 증가 |

<br>

# 5. Example

Setting

- 실수값: $$[-2.0, -1.0, 0.0, 1.0, 3.0]$$
- 표현 방식: **INT8 (0~255)** 

<br>

## (1) Symmetric Quantization

- (1) **범위**: [-max, +max] = $$[-3.0, +3.0]$$

- (2) $$\text{scale} = \frac{3.0}{127} \approx 0.0236$$.

- (3) **zero-point = 0**

<br>

양자화:

- $$q = \text{round}\left(\frac{x}{\text{scale}}\right)$$.

| **실수 값** | **정수 값 (q)** | **복원 값** |
| ----------- | --------------- | ----------- |
| -2.0        | -85             | -2.0        |
| -1.0        | -42             | -1.0        |
| 0.0         | 0               | 0.0         |
| 1.0         | 42              | 1.0         |
| 3.0         | 127             | 3.0         |

<br>

요약

- 0이 항상 정수 0에 매핑
- 범위가 0을 중심으로 대칭.

<br>

## (2) Asymmetric Quantization

- (1) **범위**: [min, max] = $$[-2.0, +3.0]$$

- (2) $$\text{scale} = \frac{3.0 - (-2.0)}{255} = \frac{5.0}{255} \approx 0.0196$$.

- (3) $$\text{zero\_point} = -\text{round}\left(\frac{-2.0}{0.0196}\right) \approx 102$$.

<br>



양자화:

- $$q = \text{round}\left(\frac{x}{\text{scale}} + \text{zero\_point}\right)$$.

| **실수 값** | **정수 값 (q)** | **복원 값** |
| ----------- | --------------- | ----------- |
| -2.0        | 0               | -2.0        |
| -1.0        | 51              | -1.0        |
| 0.0         | 102             | 0.0         |
| 1.0         | 153             | 1.0         |
| 3.0         | 255             | 3.0         |

<br>

요약

- 0이 정수 0이 아닌 **zero-point(102)**에 매핑됨
- 분포가 비대칭일 때 더 효율적

