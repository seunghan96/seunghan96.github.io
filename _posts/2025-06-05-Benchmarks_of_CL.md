---
title: Benchmarks of Continual Learning
categories: [CONT]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<br>

# Benchmarks of Continual Learning

# Contents

0. “Split” Continual Learning Benchmarks
1. Split miniImageNet-TA
2. Split miniImageNet-TF
3. CORE50-TA
4. CORE50-TF
5. Comparison

<br>

# “Split” Continual Learning Benchmarks

- **Split**: 원래의 dataset (e.g. miniImageNet, CORE50)을 **class 단위로 분할**하여 여러 개의 task를 구성
  - 각 task는 **5개의 class**로 구성됨
  - class는 **random하게, 중복 없이** 샘플링됨
- 실험 후 **총 평균 정확도(ACC)**, **망각도(FM)**, **최초 학습 정확도(LA)** 측정

<br>

## TA vs. TF

| **항목**        | **Task-Aware (TA)** | **Task-Free (TF)**     |
| --------------- | ------------------- | ---------------------- |
| Task ID 사용    | O                   | X                      |
| Softmax head    | task별 분리 가능    | 통합 softmax head 필수 |
| Class confusion | X                   | O                      |
| 현실성          | $$\downarrow$$      | $$\uparrow$$           |
| 난이도          | $$\downarrow$$      | $$\uparrow$$           |

![figure2](/assets/img/CONT/img30.png)

- https://arxiv.org/pdf/2403.05175

<br>

# 1. Split miniImageNet-TA

| **항목**            | **설명**                                                     |
| ------------------- | ------------------------------------------------------------ |
| Dataset             | miniImageNet (100 classes 중 일부 사용)                      |
| Split 방식          | 총 **20개 task**로 분할 (3 validation, 17 continual tasks)   |
| Task 구성           | 각 task는 5개의 새로운 class                                 |
| **TA (Task-Aware)** | 모델은 **현재 task 번호를 알고 있음** → 해당 task 전용 classifier 사용 가능 |
| 평가 방식           | 매 task 후, 그 task에만 맞는 softmax head로 정확도 평가      |

- **쉬운 설정**
- Task 정보가 주어지므로 **task confusion 없음**

<br>

## Details

(1) Dataset 세 줄 요약

- ImageNet에서 추출한 **소형 데이터셋**으로,
- 보통 **100개 클래스**, 클래스당 600장 이미지 (84×84 RGB)로 구성
- **few-shot learning, meta-learning, continual learning** 등에서 널리 사용됨

<br>

(2) Split 관련

- miniImageNet의 100개 클래스를 **task 단위로 쪼갠 것**
- e.g., 100개 클래스를 5개씩 나누면..
  - **20개의 task**
  - 각 task는 **5-way classification** 문제

<br>

(3) Task-Aware (TA)

- 모델이 **“지금 내가 어떤 task를 학습 중인지 알고 있음”**을 의미함

- 즉, 학습/테스트 시 **task ID가 제공**되므로...

  $$\rightarrow$$ 그 task에 해당하는 **클래스들만 분류하도록 softmax head를 제한**할 수 있음

<br>

(4) 구성 예시

| **Task** | **Classes** |
| -------- | ----------- |
| Task 1   | class 0~4   |
| Task 2   | class 5~9   |
| …        | …           |
| Task 20  | class 95~99 |

- 각 task는 독립적인 5-way 분류 문제
- 모델은 Task 2의 경우, “class 5~9 중에서 정답을 골라야 함”을 알고 있음 (TA 이므로!)

<br>

(5) 평가 방식

- 모델은 task N의 test set을 볼 때 **task N 전용 classifier만 활성화**
- 즉, **softmax head는 5개만 활성화**되고 나머지는 mask됨

<br>

# 2. Split miniImageNet-TF

| **항목**           | **설명**                                                     |
| ------------------ | ------------------------------------------------------------ |
| Dataset            | miniImageNet                                                 |
| Split 방식         | 위와 동일하게 20 task 구성                                   |
| **TF (Task-Free)** | 모델은 **task ID를 모름** → 전체 학습한 class에 대해 softmax 예측해야 함 |
| 평가 방식          | 전체 softmax over seen classes                               |

- **현실에 더 가까운 어려운 설정**
- 모델은 task가 바뀐 줄 모르고 **누적된 전체 class 중에서 예측**
- Task confusion 발생 가능 → **head를 나눌 수 없음**

<br>

## Details

(1) Dataset 세 줄 요약: 동일

(2) Split 관련: 동일

(3) Task-Free (TF)

- 모델이 **“지금 내가 어떤 task를 학습 중인지 모름”**

- 즉, 학습/테스트 시 **task ID가 주어지지 않음**

  → 모델은 지금이 어떤 task인지 **추측 없이 전부 예측**해야 함

  $$\rightarrow$$ 지금까지 **학습된 모든 클래스에 대해 softmax를 수행**

<br>

(4) 구성 예시

| **Task** | **Classes** |
| -------- | ----------- |
| Task 1   | class 0~4   |
| Task 2   | class 5~9   |
| …        | …           |
| Task 20  | class 95~99 |

- TA와 마찬가지로, 각 task는 여전히 독립적인 5-way 분류 문제이지만...
- TF 설정에서는 **"task 구분 없이"** 누적된 **"전체"** 클래스에 대해 예측!
  -  e.g., Task 3의 테스트에서 → class 0~14 전체 중에서 정답을 골라야 함

<br>

(5) 평가 방식

- 모델은 매 task의 test set에 대해 **누적된 전체 클래스에 대해 softmax 수행**
- task 1 → 5-way softmax
- task 2 → 10-way softmax
- task 20 → 100-way softmax

$$\rightarrow$$ task가 지날수록 어려워짐!

<br>

# 3. CORE50-TA

| **항목**   | **설명**                                                     |
| ---------- | ------------------------------------------------------------ |
| Dataset    | CORE50 (50개 object class + 다양한 변형)                     |
| Split 방식 | **13 task (3 validation + 10 continual)**로 구성             |
| Task 구성  | 각 task는 5개 class                                          |
| Task-Aware | task 번호가 제공됨 → 해당 task에만 맞는 classifier 사용 가능 |

- 모델은 매 task마다 정확히 분리된 softmax classifier 사용 가능
- 현실보단 이상적인 환경이지만, **object recognition처럼 복잡한 시각 데이터**에 적합

<br>

# 4. CORE50-TF

| **항목**   | **설명**                                                     |
| ---------- | ------------------------------------------------------------ |
| Dataset    | CORE50                                                       |
| Split 방식 | 동일하게 13개의 task로 나눔                                  |
| Task-Free  | task 번호 제공되지 않음 → 전체 softmax head로 누적된 class 중 예측해야 함 |

- 가장 **challenging**한 설정 중 하나
- object recognition + no task ID → **forgetting + confusion 모두 발생**

<br>

# 5. Comparison

(Feat. ChatGPT)

| **Benchmark**         | **Domain**         | **Split** | **Task Info** | **난이도** | **용도**        |
| --------------------- | ------------------ | --------- | ------------- | ---------- | --------------- |
| Split miniImageNet-TA | ImageNet subset    | 20 tasks  | ✅ 제공        | ★★☆☆☆      | 구조 파악 실험  |
| Split miniImageNet-TF | ImageNet subset    | 20 tasks  | ❌ 없음        | ★★★★☆      | realistic CL    |
| CORE50-TA             | Object recognition | 13 tasks  | ✅ 제공        | ★★★☆☆      | real-data 기반  |
| CORE50-TF             | Object recognition | 13 tasks  | ❌ 없음        | ★★★★★      | hardest setting |
