---
title: (Quantization) (4) Q-LoRA
categories: [LLM, MULT]
tags: []
excerpt: Q-LoRA (Quantized Low-Rank Adaptation)
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Contents

1. LoRA vs. Q-LoRA
2. Quantization of Q-LoRA
3. NF4 (NormalFloat 4-bit) 
4. INT4 vs. NF4 

<br>

# 1. LoRA vs. Q-LoRA

**LoRA**: 

- LLM의 **weight "전체"를 fine-tuning하지 않고**
- weight를 **저차(rank-r) "행렬 분해" 형태**로 **"추가 학습"**

**Q-LoRA**: 

- 더 나아가 **기존 weight(동결된 부분)를 quantization** 
- GPU 메모리를 줄이기 위해!

<br>

### Procedure

1. **기존 weight**
   - **"4-bit NF4"** quantization (INT4 기반, NormalFloat 포맷)
   - 메모리 절약: 원래 FP16 대비 4배 줄어듦.
   
2. **LoRA 어댑터**
   - 작은 랭크의 추가 행렬을 **"FP16/BF16"**로 학습.
   - gradient update는 여기서만 발생.
   
3. **Training/Inference**
   - Quantized weight (INT4) → Dequantize (스케일링 복원)
   
     - LoRA 어댑터 (FP16) 합쳐서 forward pass 진행.
   

<br>

# 2. Quantization of Q-LoRA

### **4-bit NormalFloat (NF4)** quantization

[세 줄 요약]

- Q-LoRA 논문 (*Dettmers et al., 2023*)에서 제안

- **INT4** 과 같은 방식!

- INT4와의 차이점?

  - INT4: 단순한 -8~7 같은 **uniform INT4**
  
  
    - NF4: 실수 분포(Gaussian-like)에 맞춰서 **“정규화된 분포를 4비트 integer로 매핑”**하는 INT4 변형
  

<br>

# 3. NF4 (NormalFloat 4-bit)

## (1) 기본 아이디어

Motivation

- **INT4**: 단순히 -8 ~ +7 같은 균일한 격자 (uniform grid)를 사용
- **LLM의 weight 분포**: 보통 **0 중심의 Gaussian distn** 모양

$$\rightarrow$$ 그래서 INT4처럼 **균등하게 쪼개면**, weight 대부분이 몰려 있는 구간(0 근처)을 제대로 표현하지 못함

$$\rightarrow$$ ***희소한 큰 값들에 너무 많은 비트를 낭비***

<br>

## (2) 작동 원리

NF4는 총 **16개의 값(4비트)**만 표현할 수 있다는 제약은 동일

$$\rightarrow$$ 다만, 이를 **정규분포의 분위수(quantile)**에 맞춰 배치!

<br>

Summary

- 즉, **"각 bin"**이 weight 분포에서 **"동일한 확률 질량 (probability mass)"**을 가지도록 설계

- 결과: weight가 ***실제 분포에서 차지하는 빈도에 비례***해 더 정밀하게 근사됨.

<br>

## (3) 공식적 설명

- [가정] $$\theta_{\text{LLM}}$$~  $$N(0,1)$$

- $$-\infty부터 +\infty$$까지 **cdf를 16등분** 

  → 각 분위수 지점 선택

  $$\rightarrow$$ ***이 16개의 값이 NF4에서 표현 가능한 대표 값***

- 실제 weight는 **scale factor로 정규화된 뒤**, 이 대표 값 중 가장 가까운 값으로 mapping

<br>

## (4) Example

- INT4: $$[-8, -7, …, 0, …, +7]$$ → 간격이 균일.

- NF4: $$[-3.5, -2.1, -1.2, -0.7, -0.3, -0.1, 0, 0.1, 0.3, 0.7, 1.2, 2.1, 3.5]$$

  → 0 근처에 더 많은 표현점이 있어서 **작은 weight 차이도 잘 보존**.

<br>

# 4. INT4 vs. NF4 

| **항목**  | **INT4**                            | **NF4 (NormalFloat 4-bit)**             |
| --------- | ----------------------------------- | --------------------------------------- |
| 격자 배치 | 균등 간격 (-8~7)                    | 정규분포 분위수 기반 (비선형)           |
| 표현 중심 | 값 범위를 고르게 표현               | 0 근처(자주 등장하는 weight)에 집중     |
| 손실 특성 | 작은 weight 손실 큼                 | 작은 weight 보존 ↑, 큰 값은 거칠게 근사 |
| 적용 사례 | 일반적인 post-training quantization | Q-LoRA에서 기본 사용                    |
| 효과      | 단순, 빠름                          | 동일 4bit에서도 정확도 유지 ↑           |

