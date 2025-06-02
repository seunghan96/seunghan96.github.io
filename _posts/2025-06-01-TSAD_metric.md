---
title: Time Series Anomaly Detection (TSAD) metric
categories: [TS, ML]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


## TSAD metric들

<br>

# 1. Precision, Recall, F1-score

| **실제 / 예측**     | **Positive (이상 탐지)** | **Negative (정상)** |
| ------------------- | ------------------------ | ------------------- |
| **Positive (이상)** | TP (True Positive)       | FN (False Negative) |
| **Negative (정상)** | FP (False Positive)      | TN (True Negative)  |

$\text{Precision} = \frac{TP}{TP + FP}$

$\text{Recall} = \frac{TP}{TP + FN}$

$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

<br>



# 2. AUROC, AUPRC

a.k.a

| **AUROC** | Area Under the ROC Curve              | = **ROC-AUC** |
| --------- | ------------------------------------- | ------------- |
| **AUPRC** | Area Under the Precision-Recall Curve | = **PR-AUC**  |

모델이 **score 기반으로 이상을 얼마나 잘 구분하는지**를 측정하는 **임계값(threshold)-무관 평가 지표**

- ROC = Receiver Operating Characteristic 

<br>

AUROC

- (X,y) = (FPR, TPR=Recall)
- 1일수록 good
  - 완전 무작위 분류기의 경우 AUC ≈ 0.5

- **Class imbalance에 약함** → 이상 탐지에선 주로 PR-AUC 선호

<br>

AUPRC

- (X,y) = (Recall, Precision)
- 1일수록 good
  - 완전 무작위 분류기의 경우 AUC ≈ 0.5

- **Class imbalance에 적함**

<br>

# 3. AUC curve 그리기

## Step 1. **이상 점수(anomaly score)** 예측

DL 모델이 각 시점별로 예측한 이상 점수 (continuous value, 예: reconstruction error, 예측 오차 등)

```python
y_score = [0.1, 0.2, 0.95, 0.9, 0.15, 0.05]
```

<br>

## Step 2. 정답 레이블 (ground truth) 불러오기

각 시점이 정상(0)인지 이상(1)인지 나타냄

```py
y_true = [0, 0, 1, 1, 0, 0]
```

<br>

## Step 3-a) (w/ Package)

AUC curve

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()
```

<br>

PR curve

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_true, y_score)
pr_auc = average_precision_score(y_true, y_score)

plt.figure()
plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.show()
```

<br>

## Step 3-b) (w/o Package)

**이상 점수(anomaly score)** 에 대해 **여러 임계값(threshold)** 을 시도

$\rightarrow$ **각 threshold에서의 분류 성능 (TP, FP 등)** 을 계산해 그리는 **곡선**

<br>

Procedure

1. y_score를 **내림차순 정렬**해서 모든 점수를 threshold처럼 생각함
2. 각 threshold에 대해 y_score >= threshold면 이상이라고 간주
3. 이에 따라 **TP, FP, FN, TN 계산**
4. 각각의 **(FPR, TPR)** 또는 **(Recall, Precision)** 을 좌표로 찍음
5. 그 좌표들을 선으로 연결하면 ROC / PR Curve 완성

<br>

# 4. Application

### 상황 1:

**“일반 유저를 실수로 핵유저로 오탐하는 위험이 매우 커.”**

- **정상인데 이상으로 잘못 탐지(FP)** 하면 큰 문제임

  → **False Positive를 줄이는 게 핵심**

- 필요한 지표: **Precision (정밀도)**

<br>

### 상황 2:

**“단 하나의 이상치도 놓치면 안 된다.”**

- **이상인데 정상으로 놓치면(FN)** 치명적임

  → **False Negative를 줄이는 게 핵심**

- 필요한 지표: **Recall (재현율)**

