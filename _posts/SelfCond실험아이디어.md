SelfCond를 제안할 때 수행할 수 있는 분석들을 체계적으로 정리해드리겠습니다: 

## 📊 1. 기본 성능 비교 (Core Performance Analysis)

### 1.1 Baseline 대비 성능 향상
- **vs. No Self-Conditioning**: SelfCond 유무에 따른 직접 비교
- **vs. Other Methods**: 다른 probabilistic forecasting 방법들과 비교 (TimeGrad, CSDI, SSSD 등)
- **Metrics**: CRPS, MAE, MSE, PICP, QICE, QL

### 1.2 데이터셋별 분석
- **Dataset Characteristics에 따른 효과**:
  - Small vs. Large datasets (데이터 크기)
  - Low-dim vs. High-dim (변수 개수)
  - Stationary vs. Non-stationary (시계열 특성)
  - Smooth vs. Volatile (변동성)

### 1.3 Prediction Horizon별 성능
- Short-term (24, 48 steps) vs. Long-term (96, 192, 336, 720 steps)
- Horizon이 길어질수록 SelfCond의 효과가 어떻게 변하는지

---

## 🔬 2. Ablation Studies

### 2.1 Self-Conditioning Probability (p) 분석
- **Grid search**: p ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0}
- **Optimal p 찾기**: 데이터셋별로 최적 확률이 다른지
- **Trade-off 분석**: p=1.0 (항상 사용) vs. p=0.5 (확률적 사용)

### 2.2 Self-Conditioning Source 비교
- **v1**: Random noise
- **v2**: Model prediction (same timestep)
- **v3**: Model prediction (previous timestep t+1→t)
- **각 버전의 장단점과 적용 시나리오**

### 2.3 Diffusion Steps 분석
- **T ∈ {10, 20, 30, 40, 50, 100}**
- SelfCond가 적은 step에서 더 효과적인지
- Sampling efficiency vs. Quality trade-off

### 2.4 Model Capacity (d_model)
- **d ∈ {128, 256, 512, 1024}**
- 작은 모델에서 SelfCond의 regularization 효과
- 큰 모델에서의 추가 이득

---

## 📈 3. 학습 과정 분석 (Training Dynamics)

### 3.1 Convergence 속도
- **Learning curves**: SelfCond vs. Baseline
- **Epochs to convergence**: 더 빨리 수렴하는가?
- **Training stability**: Loss의 variance가 줄어드는가?

### 3.2 Overfitting 분석
- **Train vs. Val gap**: SelfCond가 regularization 역할을 하는가?
- **Early stopping point**: 언제 멈추는 것이 최적인가?

### 3.3 Gradient 분석
- **Gradient norm**: 안정적인가?
- **Gradient flow**: Vanishing/exploding 문제가 완화되는가?

---

## 🎯 4. 예측 품질 분석 (Prediction Quality)

### 4.1 Calibration 분석
- **Prediction Interval Coverage Probability (PICP)**
  - Target: 95% → 실제로 95%에 가까운가?
- **Quantile Calibration**
  - 각 quantile (0.1, 0.3, 0.5, 0.7, 0.9)의 정확도
- **QICE (Quantile Interval Coverage Error)**

### 4.2 Sharpness vs. Calibration
- **Interval width 분석**: 예측 구간이 좁으면서도 정확한가?
- **Sharpness score**: CRPS가 낮으면서 PICP도 좋은가?

### 4.3 Distributional Metrics
- **Energy Score**: 전체 분포의 정확도
- **Variogram Score**: Multivariate dependency 포착
- **Quantile Score**: 특정 quantile의 정확도

---

## 🔍 5. 시각화 분석 (Visual Analysis)

### 5.1 예측 분포 시각화
- **Fan chart**: Quantile bands (10%, 30%, 50%, 70%, 90%)
- **Spaghetti plot**: 여러 샘플들의 trajectory
- **Baseline vs. SelfCond 비교**: 더 날카롭고 정확한가?

### 5.2 Case Study
- **Best cases**: SelfCond가 크게 개선한 예시
- **Worst cases**: 오히려 나빠진 예시 (왜?)
- **Failure analysis**: 어떤 패턴에서 실패하는가?

### 5.3 Uncertainty 분석
- **Epistemic vs. Aleatoric uncertainty**
- **Time-varying uncertainty**: 예측 horizon에 따른 불확실성 증가
- **Event-based uncertainty**: 특정 이벤트에서의 불확실성

---

```
File "/home/seunghan9613/NsDiff-main-v2/transfer_learning/train_source_only.py", line 88, in train_source_model
    print(f"Model weights saved in: {experiment.run_save_dir}")
AttributeError: 'NsDiffForecast' object has no attribute 'run_save_dir'
```



## ⚡ 6. 효율성 분석 (Efficiency)

### 6.1 Training 효율성
- **Training time**: SelfCond가 느린가? (약 8% overhead 예상)
- **Memory usage**: 추가 메모리가 필요한가?
- **Computational cost**: FLOPs 비교

### 6.2 Inference 효율성
- **Sampling time**: 추론 속도 비교
- **Number of samples needed**: 같은 품질에 필요한 샘플 수
- **Reduced sampling steps**: SelfCond로 step 수를 줄일 수 있는가?

### 6.3 Scalability
- **Large datasets**: Traffic (862 dims), Electricity (321 dims)
- **Long sequences**: 시퀀스 길이 증가 시 효과

---

## 🧪 7. Robustness 분석

### 7.1 Noise Robustness
- **Input noise**: 입력에 노이즈를 추가했을 때
- **Missing data**: 결측치가 있을 때
- **Outliers**: 이상치에 강건한가?

### 7.2 Distribution Shift
- **Train-test mismatch**: 분포가 달라졌을 때
- **Concept drift**: 시간에 따른 패턴 변화
- **Transfer learning**: 다른 도메인으로 전이

### 7.3 Hyperparameter Sensitivity
- **p 변화에 따른 robustness**
- **T 변화에 따른 robustness**
- **Initialization sensitivity**

---

## 🎲 8. Stochastic Behavior 분석

### 8.1 Sample Diversity
- **Inter-sample variance**: 생성된 샘플들의 다양성
- **Mode coverage**: 멀티모달 분포를 잘 포착하는가?
- **Collapse analysis**: Mode collapse가 발생하는가?

### 8.2 Seed Stability
- **Multiple seeds**: 여러 seed로 실험 (1, 2, 3, 42, 100)
- **Variance across seeds**: 결과가 안정적인가?
- **Confidence intervals**: 평균 ± 표준편차

---

## 🔄 9. Iterative Refinement 분석

### 9.1 Self-Conditioning의 Quality Evolution
- **t+1 → t 과정에서 prediction quality 개선 추적**
- **몇 step에서 가장 큰 개선이 있는가?**
- **Early vs. Late timesteps의 기여도**

### 9.2 Self-Consistency
- **ŷ₀^prev와 ŷ₀의 일관성 측정**
- **Consistency가 성능과 상관관계가 있는가?**

---

## 📐 10. 이론적 분석 (Theoretical Analysis)

### 10.1 Loss Landscape
- **Loss surface visualization**: SelfCond가 더 smooth한가?
- **Local minima**: 더 좋은 minima에 도달하는가?

### 10.2 Posterior Approximation
- **KL divergence**: True posterior와의 차이
- **Evidence Lower Bound (ELBO)**: 더 tight한 bound인가?

### 10.3 Information Flow
- **Mutual information**: ŷ₀^prev와 y₀의 상호정보
- **Information bottleneck**: 어디서 정보 손실이 발생하는가?

---

## 🎯 11. Application-Specific 분석

### 11.1 Decision Making
- **Risk-sensitive forecasting**: 극단값 예측
- **Cost-sensitive metrics**: 비용 함수 기반 평가
- **Action recommendation**: 예측 → 의사결정

### 11.2 Anomaly Detection
- **Likelihood-based**: 낮은 likelihood = 이상치
- **Reconstruction error**: SelfCond가 정상 패턴을 더 잘 학습하는가?

---

## 📊 12. Comparative Studies

### 12.1 Other Conditional Methods
- **Guidance**: Classifier-free guidance vs. SelfCond
- **Conditioning mechanisms**: Cross-attention vs. Concat vs. SelfCond

### 12.2 Other Diffusion Improvements
- **DDIM vs. SelfCond-DDPM**
- **Classifier-free guidance + SelfCond**: 시너지 효과

---

## 🎨 추천 분석 우선순위

### **Tier 1 (필수)**
1. ✅ Baseline vs. SelfCond 성능 비교 (모든 datasets)
2. ✅ Ablation: p (self-conditioning probability)
3. ✅ v1 vs. v2 vs. v3 비교
4. ✅ Calibration 분석 (PICP, QICE)
5. ✅ 시각화 (fan charts, case studies)

### **Tier 2 (중요)**
6. Training efficiency (time, convergence)
7. Hyperparameter grid search (d_model, T, p)
8. Robustness (noise, missing data)
9. Prediction horizon 분석
10. Sample diversity 분석

### **Tier 3 (부가)**
11. Theoretical analysis
12. Transfer learning
13. Anomaly detection
14. Loss landscape visualization

---

이 중에서 **어떤 분석을 구현**하고 싶으신가요? 구체적인 코드를 작성해드리겠습니다! 🚀