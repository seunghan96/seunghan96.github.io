좋아요. Figure 1의 세 그림은 다음 세 단계를 보여줍니다 :

1️⃣ **상단:** 기본 DDPM (endpoint: N(0, I)) — 평균과 분산이 모두 고정된 **stationary** 확률 모델.

2️⃣ **중간:** TMDM (endpoint: N(f(X), I)) — 평균은 X에 따라 달라지지만, 분산은 여전히 고정.

3️⃣ **하단:** NsDiff (endpoint: N(f(X), g(X))) — 평균과 분산 모두 X에 따라 달라지는 **Location-Scale Noise Model(LSNM)** 기반 비정상(diffusion) 모델.



즉, 고도화 방향은

**(고정 평균·분산) → (동적 평균) → (동적 평균 + 동적 분산)**

의 3단계입니다.



------





### **🚀 이를 뛰어넘는 novel idea 제안 (NsDiff 이후 단계)**







#### **💡 개념 이름 제안:** 

#### **Adaptive Dynamic Diffusion (AD²)**



**핵심 아이디어:**

NsDiff가 입력 X에 따라 **mean (f(X))**과 **variance (g(X))**를 결정했다면, 그 다음 단계는 **시간적 변화와 구조적 불확실성 자체를 확률적으로 학습**하는 것입니다. 즉, g(X)의 deterministic mapping을 넘어, **uncertainty dynamics itself**를 모델링합니다.

1. **Uncertainty Process Diffusion (UPD)**

   

   - 기존 NsDiff는 g(X)를 고정된 함수로 본다.

   - AD²에서는 g(X, t)를 **latent stochastic process**로 정의.

   - forward diffusion에서 noise level βₜ를 g(X, t)에 의해 *sampling adaptive*하게 조정.

   - 결과적으로, 모델이 “불확실성의 시간적 진화”까지 학습하게 됨.

   - 식: Y = f(X, t) + \sqrt{g(X, t)} \, \epsilon, \quad g(X, t) \sim \text{DiffusionProcess}(ϕ)

   
   
   
2. **Cross-scale Uncertainty Coupling**

   - g(X) 대신 **multi-scale variance field gₛ(X)**를 도입.

   - temporal scale s마다 불확실성이 다르게 작용 (예: 단기 noise vs 장기 trend).

   - forward process에서 βₜ를 scale-weighted 합으로 구성:

     \beta_t = \sum_s w_s \, g_s(X).

   - 이렇게 하면 uncertainty propagation이 시간적 resolution에 따라 다르게 작용.








1. **Bayesian Diffusion Scheduling**

   RuntimeError: mat1 and mat2 shapes cannot be multiplied (5376x5 and 4x128)

   - uncertainty-aware noise schedule을 deterministic하게 주지 않고, posterior 분포로 샘플링.

   - 즉, \beta_t \sim p(\beta|X),

     예: \beta_t \sim \mathcal{N}(μ_β(X), σ_β(X)).

   - 이는 variance 예측 오차(gψ bias)를 스스로 보정하는 meta-level uncertainty estimation.

   

2. **Semantic Uncertainty Embedding**

   - variance g(X)를 직접 예측하는 대신, **feature embedding 공간에서 불확실성 구조**를 학습.

- 즉, variance field를 latent representation z(X)에서 추론: g(X)=Decoder(z(X)).
   - z(X)는 temporal attention으로 요약된 “uncertainty context”.
   
   

5. **Hybrid Frequency-Aware NsDiff**

   - variance g(X)를 **frequency domain**에서도 추정 (저주파: trend variance, 고주파: noise variance).

   - time–freq uncertainty fusion:

     g(X) = λ_t g_t(X) + λ_f g_f(FFT(X)).

   - frequency-dependent noise schedule은 계절성 불확실성을 더 잘 캡처.

   

en(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: './results/runs/F/ILI/w168h1s36/1/best_mode



![image-20251023192319018](/Users/seunghan96/Library/Application Support/typora-user-images/image-20251023192319018.png)
