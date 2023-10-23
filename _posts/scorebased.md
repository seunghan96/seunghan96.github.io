# Score-based Generative Modeling by Diffusion Process

# 1. Intro to Generative Models

Examples of Generative Model:

- (1) Autoencoder (AE) ( + VAE )

- (2) GAN

- (3) Normalizing Flows
- **(4) SCORE-BASED GENERATIVE MODELS**

<br>

![figure2](/assets/img/ts/img450.png)

<br>

# 2. Limitations of Generative Models

## a) Categories of Generative Models

[1] Likelihood-based : Autoencoder (AE) ( + VAE ), Normalizing Flows

[2] GAN-based: GAN

<br>

## b) Limitations

[1] Likelihood-based

- need specific architecture ( less flexible )
  - ex) Autoregressive model, Flow model
- VAE: Surrogate loss ( ex. ELBO )

<br>

[2] GAN-based: unstable training

<br>

# 3. Gerative Modeling by Estimating Gradients of the Data Distribution

## (1) Score

Score

= **(1) Gradient** of **(2) Log likelihood of Data**

= $\nabla_{\mathbf{x}} \log p(\mathbf{x})$.

<br>

주의할 점:

- **parameter에 대한 score**가 아니다 : $-\nabla_\theta \log (\mathcal{L}(\theta))$ (X)
- **data에 대한 score** 이다! : $\nabla_{\mathbf{x}} \log p_{\text {data }}(\mathbf{x})$ (O)

<br>

## (2) Score Matching

**전체 data**에 대한 score를 계산은 불가능!

**주어진 data**에 대한 score를 계산하고, **전체 data**에 대해 추정해야

$\rightarrow$ need ***Score Estimator***

**( = 임의의 data에 대해 score를 추정하는 모델 )**

<br>

Score Estimator, $\mathbf{s}_\theta(x)$

- **score estimator** = score을 추정하는 모델

- **score matching**  = score estimator를 학습하는 방법

- **score matching**의 objective function:

  - $\mathcal{L}_\theta=\mathbb{E}_{p_{\text {data }}}\left\|\mathbf{s}_\theta(\mathbf{x})-\nabla_{\mathbf{x}} \log p_{\text {data }}(\mathbf{x})\right\|_2^2$.

    ( = **(1) 실제 score** & **(2) 추정 score** 간의 MSE )

<br>

### Question) 

$p_{\text {data }}(\mathbf{x})$ 를 모르는데, **(1) 실제 score**는 어떻게 알지?

<br>

### Answer) 

위의 Loss function을, $p_{\mathrm{data}}(\mathbf{x})$에 의존하지 않도록 수식 유도 가능!

- (before) $\mathcal{L}_\theta=\mathbb{E}_{p_{\text {data }}}\left\|\mathbf{s}_\theta(\mathbf{x})-\nabla_{\mathbf{x}} \log p_{\text {data }}(\mathbf{x})\right\|_2^2$
- (after) $\mathcal{L}_\theta=\mathbb{E}_{p_{\text {data }}(\mathbf{x})}\left[\operatorname{tr}\left(\nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x})\right)+\frac{1}{2}\left\|\mathbf{s}_\theta(\mathbf{x})\right\|_2^2\right]$

<br>

### Problem?

문제점: $\nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x})$에 대한 계산이 high-dim 상황에서 쉽지 않다!

해결책: ***DENOISING*** score matching

<br>

## (3) Denoising Score Matching

(a) 아이디어: $\operatorname{tr}\left(\nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x})\right)$에 대한 계산을 피하자!

(b) 핵심: 

- 원본 data에 대한 score를 계산 (X)
- perturbed data에 대해 score 계산 (O)

(c) 방법:

- 미리 정의된 noise 분포 $q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})$ 를 이용 
- perturbed data distn:
  - $q_\sigma(\tilde{\mathbf{x}}) \triangleq \int q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) p_{d a t a}(\mathbf{x}) \mathrm{d} \mathbf{x}$.

<br>

Loss function:

- (before) $\mathcal{L}_\theta=\mathbb{E}_{p_{\text {data }}(\mathbf{x})}\left[\operatorname{tr}\left(\nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x})\right)+\frac{1}{2}\left\|\mathbf{s}_\theta(\mathbf{x})\right\|_2^2\right]$
- (after) $\mathcal{L}_\theta=\mathbb{E}_{q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) p_{\text {data }}}\left[\left\|\mathbf{s}_\theta(\tilde{\mathbf{x}})-\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})\right\|_2^2\right]$

<br>

한 줄 요약 : **원본 data distn** 대신 **perturbed data distn** 사용하자!!

( perturbed data distn에 대한 density는 직접 계산 가능하므로! )

<br>

## (4) Sampling with Langevin Dynamics

Training & Inference ( = Generating )

- Training : by **Score Matching**
- Inference : by **Sampling**

<br>

최적 ***parameter*** 찾기 ( = SGD )

- $\theta_t=\theta_{t-1}-\eta \nabla_\theta \mathcal{L}(x ; \theta)$.

<br>

최적 ***data*** 찾기  ( = Sampling with Langevin Dynamics )

- $\tilde{\mathbf{x}}_t=\tilde{\mathbf{x}}_{t-1}+\frac{\epsilon}{2} \nabla_{\mathbf{x}} \log p\left(\tilde{\mathbf{x}}_{t-1}\right)+\sqrt{\epsilon} \mathbf{z}_t, \quad \text { where } \mathbf{z}_t \sim \mathcal{N}(0, I)$.
  - 앞서 학습한 **score estimator**를 사용해서 $\nabla_{\mathbf{x}} \log p\left(\tilde{\mathbf{x}}_{t-1}\right)$를 구함.
  - $\mathbf{z}_t$ : add RANDOMNESS

<br>

## (5) Denoising Score Matching with Langevin Dynamics (SMLD)

앞선 두 가지를 합친 것을 SMLD라고 한다!

- (3) Denoising Score Matching ( = Training )
- (4) Sampling with Langevin Dynamics ( = Inference )

<br>

***Noise Conditional Score Network***

- based on SMLD

- data를 다양한 noise로 변환

  & 변환된 data의 score를 추정!

- score network vs. NOISE CONDITIONAL score network
  - score network
    - $\mathbf{s}_\theta(\mathbf{x}) \approx \nabla_{\mathbf{x}} \log q(\mathbf{x})$.
  - NOISE CONDITIONAL score network
    - $\mathbf{s}_\theta(\mathbf{x}, \sigma) \approx \nabla_{\mathbf{x}} \log q_\sigma(\mathbf{x})$.

<br>

# 4. Denoising Diffusion Probabilistic Modeling (DDPM)

SMLD ( Noise Conditional Score Network ) 

- data distn을 noise distn으로 변환
- noise distn으로부터 data 복원

<br>

DDPM : SMLD + ***Discrete Markov Chain***

- 데이터의 noise add / remove 과정이 Discrete Markov Chain으로 표현된다.

![figure2](/assets/img/ts/img451.png)

<br>

## (1) Forward Process ( = ADD nosie )

ONE-step

- $p\left(\mathbf{x}_i \mid \mathbf{x}_{i-1}\right)=\mathcal{N}\left(\mathbf{x}_i ; \sqrt{1-\beta_i} \mathbf{x}_{x-i}, \beta_i \mathbf{I}\right)$.

MULTI-step

- $p_{\alpha_i}\left(\mathbf{x}_i \mid \mathbf{x}_0\right)=\mathcal{N}\left(\mathbf{x}_i ; \sqrt{\alpha_i} \mathbf{x}_0,\left(1-\alpha_i\right) \mathbf{I}\right), \quad \text { where } \alpha_i \triangleq \prod_{j=1}^i\left(1-\beta_j\right)$.

<br>

## (2) Backward Process ( = REMOVE noise )

( forward: Gaussian $\rightarrow$ backward: Gaussian )

ONE-step

- $p_\theta\left(\mathbf{x}_{i-1} \mid \mathbf{x}_i\right)=\mathcal{N}\left(\mathbf{x}_{i-1} ; \frac{1}{\sqrt{1-\beta_i}}\left(\mathbf{x}_i+\beta_i \mathbf{s}_\theta\left(\mathbf{x}_i, i\right)\right), \beta_i, \mathbf{I}\right)$,

<br>

아래는 같은 표현!

- Sampling 한다
- 새로운 데이터를 생성한다

- Noise에서 시작해서 Backward Process를 여러 번 반복한다
  - $\mathbf{x}_{i-1}=\frac{1}{\sqrt{1-\beta_i}}\left(\mathbf{x}_i+\beta_i \mathbf{s}_{\theta^*}\left(\mathbf{x}_i, i\right)\right)+\sqrt{\beta_i} \mathbf{z}_i, \quad i=N, N-1, \ldots, 1$.

<br>

Ancestral Sampling

= 미리 정의한 conditional distn로 구성된 $\prod_{i=1}^N p_\theta\left(\mathbf{x}_{i-1} \mid \mathbf{x}_i\right)$로 데이터를 샘플링

<br>

DDPM & SGLD

= **Score-based Generative Model**

( $\because$ 둘 다 data의 score를 간접적으로 추정! )

<br>

# 5. Score-based Generative Modeling through Stochastic Differential Equations SDEs

DDPM & SGLD의 핵심

- data를 (여러 scale의) noise로 변환
- noise로부터 data를 복원

<br>

Noise의 scale = forward step의 횟수에 따라 결정!

- step $\uparrow$ : noise $\uparrow$
- step $\downarrow$ : noise $\downarrow$

<br>

Question) What if **noise scale이 무한히 정밀??**

$\rightarrow$ ***Stochastic Differential Equations (SDE)*** 관점

- Noise를 time variable로 취급!
- (1) SDE : data 분포에 noise를 조금씩 더함
- (2) Reverse-time SDE를 : noise 분포에서 noise를 조금씩 제거함

<br>

## (1) SDE

DE (Differential Equation, 미분방정식)

- $f$ 를 모를 때, $\nabla f$ 와 $f(a)$ 를 사용해서 $f$를 추정!

SDE (Stochastic Differential Equation, 확률 미분방정식)

- DE + Stochastic Process

<br>

## (2) Perturbing Data with SDEs

Notation

- time variable: $t \in[0, T]$

- 초기값 ( = data 분포 ) : $\mathbf{x}(0) \sim p_0$
- 변환된 데이터 ( = noise 분포 ) : $\mathbf{x}(T) \sim p_T$

- time variable에 대한 데이터 : $\left\{\mathbf{x}(t)_{t=0}^T\right\}$

  ( 즉, time variable $t$에 따라 data가 update된다! = Diffusion Process )

  - data가 변화하는 정도 = $\mathrm{d} \mathbf{x}$

<br>

### Diffusion Process

$\mathrm{d} \mathbf{x}=\mathbf{f}(\mathbf{x}, t) d t+g(t) \mathrm{d} \mathbf{w}$.

- $f(\cdot, t): \mathbb{R}^d \rightarrow \mathbb{R}^d$  :  $x(t)$ 의 drift coefficient
  -  $\mathbf{x}$ 에 따른 함수값
- $g(\cdot): \mathbb{R} \rightarrow \mathbb{R}$ : diffusion coefficient
  - 해당 함수값에 얼마큼 noise를 부여할 지
  - $\mathbf{w}$ : Standard Winer Process ( for randomness )

<br>

## (3) Generating Samples by Reversing the SDE

https://blog.si-analytics.ai/49