---
title: Quantization
categories: [DLF, LLM, Python, MULT]
tags: []
excerpt: Float32 vs Float16 vs BFloat16
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Quantization

## Contents

1. Float32 vs. Float16 vs. BFloat16
2. Min & Max Range Comparison
3. Converting Data Type

<br>

# 1. Float32 vs. Float16 vs. BFloat16

## (1) Float 32

![figure2](/assets/img/llm/img496.png)

- 32 = 1 (sign) + 8 (exponent) + 23 (mantissa)
- $$(-1)^{\text {sign }} 2^{(\text {exponent }-127)} \times 1 \text {.mantissa }$$.

<br>

Example:

![figure2](/assets/img/llm/img497.png)

- $$81= 2^{6} + 2^4 + 2^0 = 64 + 16 +1$$
- (max): $$2^7 + 2^6+ \cdots 2^0 = 2^8-1 =255$$

<br>

## (2) Float16

![figure2](/assets/img/llm/img498.png)

- 16 = 1 (sign) + 5 (exponent) + 10 (mantissa)

- $$(-1)^{\text {sign }} 2^{(\text {exponent }-15)} \times 1 \text {.mantissa }$$

<br>

## (3) BFloat16

(Brain Float 16)

![figure2](/assets/img/llm/img499.png)

- 16 = 1 (sign) + 8 (exponent) + 7 (mantissa)
- $$(-1)^{\text {sign }} 2^{(\text {exponent }-127)} \times 1 . \text { mantissa }$$.

<br>

## (4) Float8

![figure2](/assets/img/llm/img500.png)

- 8 = 1 (sign) + 4 (exponent) + 3 (mantissa)
- $$(-1)^{\operatorname{sign}} 2^{\left(\text {exponent-7) } \times 1-n a-s s c_0 .\right.}$$.

<br>

# 2. Min & Max Range Comparison

Float32: $$\left[-3.4 \times 10^{38}, 3.4 \times 10^{38}\right]$$

Float16: $$\left[-6.55 \times 10^4, 6.55 \times 10^4\right]$$

BFloat16: $$\left[-3.39 \times 10^{38}, 3.39 \times 10^{38}\right]$$

Float8: $$[-240,240]$$

<br>

# 3. Converting Data Type

## (1) Float32 $$\rightarrow$$ Float16

![figure2](/assets/img/llm/img501.png)

- exponent의 **앞부분**부터
- decimal의 **뒷부분**부터

<br>

문제점? Float overflow!

- Float 16이 가질 수 있는 범위를 초과할 수도 있음!

해결책? BFloat16!

<br>

## (2) Float32 $$\rightarrow$$ BFloat16 

![figure2](/assets/img/llm/img502.png)
