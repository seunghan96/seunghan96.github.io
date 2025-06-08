---
title: Introduction to Continual Learning
categories: [CONT]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<br>

# Introduction to Continual Learning

(참고: http://dmqa.korea.ac.kr/activity/seminar/378)

<br>

# Contents

1. Transfer in CL
2. Three CL Scenarios
3. Baselines of CL
4. Categories of CL
5. Evaluation metric

<br>

# 1. Transfer in CL

- Forward transfer: 과거 task의 knowledge $$\rightarrow$$ 미래 task 성능 $$\uparrow$$
- Backward transfer: 미래 task의 knowledge $$\rightarrow$$ 과거 task 성능 $$\uparrow$$

![figure2](/assets/img/CONT/img23.png)

<br>

# 2. Three CL Scenarios

|               | Task-ID | Class             | 난이도 |
| ------------- | ------- | ----------------- | ------ |
| **Task**-IL   | O       | 현재 class        | 하     |
| **Domain**-IL | X       | 현재 class        | 중     |
| **Class**-IL  | X       | 과거 + 현재 class | 상     |

<br>

Domain-IL vs. Class-IL 예시

![figure2](/assets/img/CONT/img24.png)

<br>

# 3. Baselines of CL

(일반적) Baseline 세팅  = (Offline으로) 모든 task의 data를 동시에 사용하여 multi-task learning

<br>

Baseline의 두 종류

| **이름**                   | **설명**                              | **의미**                 |
| -------------------------- | ------------------------------------- | ------------------------ |
| **None** 또는 **Naive**    | 그냥 순차적으로 학습, 아무 방어 안 함 | **Lower bound (최저선)** |
| **Offline** 또는 **Joint** | 모든 task 데이터를 **동시에** 학습    | **Upper bound (최고선)** |

![figure2](/assets/img/CONT/img25.png)

<br>

# 4. Categories of CL

1. Regularization-based
   - Loss function + **Reg term** $$\rightarrow$$ 과거 task forgetting 방지
2. Replay-based
   - Memory buffer 안에 과거 task의 dataset을 구축 (혹은 생성)
3. Architecture-based
   - 각 Task에 대한 sub-network (or 기존 network 확장)

<br>

# 5. Evaluation metric

과거 task의 forgetting 능력을 측정하기 위한 메트릭!

![figure2](/assets/img/CONT/img26.png)
