---
title: (Quantization) (3) Scaling factor, Zero-point
categories: [LLM, MULT]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Contents

1. 기본 공식 (Uniform quantization)
2. 예시: 2×2 grid 값
   1. Scaling factor
   2. Zero-point
   3. Quantization
   4. Dequantization


<br>

# Quantization에서 scaling factor와 zero-point

<br>

# 1. 기본 공식 (Uniform quantization)

[1] **Scaling factor**

- $$\text{scale} = \frac{\text{max} - \text{min}}{\text{range}}$$

  - **max, min**: 실수(weight) 값 범위

  - **range**: 정수 표현 범위 크기 (예: INT8 → 255, INT4 → 15)


<br>

[2] **Zero-point**

- $$\text{zero\_point} = -\text{round}\left(\frac{\text{min}}{\text{scale}}\right)$$.

- 이유? (왜 음수? 왜 min?)

  $$\rightarrow$$ 이렇게 하면 **min 값 → 정수 0**에 매핑되도록 맞춰짐

<br>

[3] **양자화(quantize)**

- $$q = \text{round}\left(\frac{x}{\text{scale}} + \text{zero\_point}\right)$$.

<br>

[4] **복원(dequantize)**

- $$\hat{x} = (q - \text{zero\_point}) \times \text{scale}$$.

<br>

# 2. 예시: 2×2 grid 값

실수 weight 행렬: $$\begin{bmatrix} -1.0 & 0.0 \\ 2.0 & 3.0 \end{bmatrix}$$.

- min = $$-1.0$$

- max = $$3.0$$

- 정수 타입: **INT8 (0~255)**라고 가정 

  → range = 255

<br>

## **Step 1. Scaling factor**

$$\text{scale} = \frac{3.0 - (-1.0)}{255} = \frac{4.0}{255} \approx 0.0157$$.

<br>

## **Step 2. Zero-point**

$$\text{zero\_point} = -\text{round}\left(\frac{-1.0}{0.0157}\right) = -(-64) = 64$$.

<br>

## **Step 3. Quantization**

각 원소 $$x$$에 대해:

- $$q = \text{round}\left(\frac{x}{0.0157} + 64\right)$$.

  - $$-1.0 → q = 0$$.

  - $$0.0 → q = 64$$.

  - $$2.0 → q \approx 191$$.

  - $$3.0 → q = 255$$.

<br>

## **Step 4. Dequantization (복원)**

$$\hat{x} = (q - 64) \times 0.0157$$.

- $$0 → -1.0$$.
- $$64 → 0.0$$.
- $$191 → 약 2.0$$.
- $$255 → 3.0$$.

<br>

복원된 행렬: $$\begin{bmatrix} -1.0 & 0.0 \\ 2.0 & 3.0 \end{bmatrix}$$

