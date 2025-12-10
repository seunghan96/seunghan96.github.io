# A Study on the Evaluation of Generative Models

Metrics

- **Inception Score (IS)**
- Fréchet Inception Distance (FID)

<br>

Metric 정리 방식

1. 개념
2. 수식
3. 샘플 기반 empirical 계산 방식

<br>

# 1. **Inception Score (IS)**

- 생성된 image가 **명확한 객체**를 담고 있고, **다양한 종류의 객체**를 포함하고 있는지 평가하는 점수

- IS는 “명확성(High confidence)”과 “다양성(Diversity)”을 KL로 결합해 score로 만든 지표

<br>

## **1) 개념**

- 생성 image가

  - **(i) 뚜렷한 객체를 갖는지**

  - **(ii) 다양한 class를 포함하는지**


  를 측정하는 지표

- InceptionV3 분류기 기반

- 논문 결론: **가장 불안정하고 사용 비추천**

<br>

## **2) 수식**

- $IS = \exp\left( \frac{1}{M}\sum_{i=1}^M KL(p(y|x_g^{(i)})\|p(y)) \right)$.

<br>

## **3) Empirical 계산**

- 생성 샘플 $x_g^{(i)}$을 Inception V3에 넣어 $p(y|x_g^{(i)})$ 계산

- 전체 marginal:
  - $p(y)=\frac{1}{M}\sum_i p(y|x_g^{(i)})$.

- Exponential 씌우기

<br>

## 4) Details

**Step 1) 생성된 image마다 InceptionV3 분류기에 forward**

- i.e., 생성된 image 하나 $x_g^{(i)}$가 들어가서 $p(y|x_g^{(i)})$가 나옴
- 의미 = **“이 image가 어떤 class ($i$)일 확률인가?”**를 알려주는 distn

- ```
  dog: 0.9
  cat: 0.05
  car: 0.02
  ...
  ```

- 이렇게 **한 class에 확실하게 쏠려 있으면** → image가 명확하고 품질 좋다는 뜻.

<br>

**Step 2)  전체 생성 image들의 “class 분포”를 평균**

- for **다양성 측정**
- 모든 생성 image에 대해 평균을 내면:
  - $p(y) = \frac{1}{M}\sum_i p(y|x_g^{(i)})$.

<br>

Example) 전체 생성 image가

- 대부분 개(dog)만 나오면 → $p(y)$가 dog에 몰림 → **다양성 낮음**
- 다양한 동물/사물/풍경이 나오면 → $p(y)$가 고르게 퍼짐 → **다양성 높음**

<br>

**Step 3) 이제 KL divergence로 “명확성 + 다양성”을 합쳐 평가**

- $KL(p(y|x) \,\|\, p(y))$.
- 해당 값이 크다는 의미?
  - **$p(y|x)$** = 개별 image의 class 분포가 한 class에 강하게 쏠려 있고
  - **$p(y)$** = 전체 class 분포는 균등함

<br>

**Step 4) Exp 감싸기**

- 전체 평균을 exp로 감싸면 IS Score
  - for 단순히 값의 scale을 키워주기
- $IS = \exp\left( \frac{1}{M}\sum_i KL(p(y|x_i) \,\|\, p(y)) \right)$.

<br>

## 5) Example (잘한/못한 케이스)

$IS = \exp\left( \frac{1}{M}\sum_{i=1}^M KL\big(p(y|x_i)\,\|\,p(y)\big) \right)$.

- where $p(y)=\frac{1}{M}\sum_{i=1}^M p(y|x_i)$.

<br>

Notation

- 생성한 Image 개수 ($M$) = 2
- $y\in\{\text{cat},\text{dog}\}$.

<br>

### (a) Good

생성한 2개의 image의 pred prob

- $p(y|x_1) = (0.9,\,0.1)$

- $p(y|x_2) = (0.1,\,0.9)$.

$\rightarrow$ $p(y) = \frac{1}{2}\big[(0.9,0.1) + (0.1,0.9)\big] = (0.5,\,0.5)$

<br>

$KL_1 = 0.9\log\frac{0.9}{0.5} + 0.1\log\frac{0.1}{0.5} \approx 0.3681$.

$KL_2 = 0.1\log\frac{0.1}{0.5} + 0.9\log\frac{0.9}{0.5} \approx 0.3681$.

$\rightarrow$ $\overline{KL} = \frac{KL_1 + KL_2}{2} \approx 0.3681$

<br>

$IS_{\text{good}} = \exp(\overline{KL}) \approx \exp(0.3681) \approx 1.45$.

<br>

### (b) Bad

생성한 2개의 image의 pred prob

- $p(y|x_1) = (0.9,\,0.1)$.
- $p(y|x_2) = (0.8,\,0.2)$.

$\rightarrow$ $p(y) = \frac{1}{2}\big[(0.9,0.1) + (0.8,0.2)\big] = (0.85,\,0.15)$.

<br>

$KL_1 = 0.9\log\frac{0.9}{0.85} + 0.1\log\frac{0.1}{0.15} \approx 0.0109$.

$KL_2 = 0.8\log\frac{0.8}{0.85} + 0.2\log\frac{0.2}{0.15} \approx 0.0089$.

$\rightarrow$ $\overline{KL} = \frac{KL_1 + KL_2}{2} \approx 0.0099$

<br>

$IS_{\text{bad}} = \exp(\overline{KL}) \approx \exp(0.0099) \approx 1.01$.

<br>

# 2. **FID (Fréchet Inception Distance)**

- 생성된 image 분포와 실제 image 분포가 **얼마나 가까운지**를 측정
- InceptionV3 feature 공간에서 **두 분포의 mean & cov 차이**를 프레셰 거리(Fréchet Distance)로 계산  

<br>

## **1) 개념**

FID는 두 분포를 비교

- **Real image 분포**: 
  - **실제** dataset image를 **"InceptionV3 feature"**로 임베딩한 분포  

- **Generated image 분포**: 
  - **생성된** image를 **"InceptionV3 feature"**로 임베딩한 분포  


<br>

이 두 분포의  

- **평균 (μ)**  
- **공분산 (Σ)**  

을 비교해 **얼마나 겹치는지** 측정

<br>

→ 값이 **작을수록** 두 분포가 비슷하다 = **더 좋은 생성 품질**

<br>

핵심 평가 요소

- **Quality**: Real과 유사한 feature 표현을 가지는가?
- **Diversity**: 분포 전체가 real data를 잘 커버하는가?

<br>

## **2) 수식**

$FID = \|\mu_r - \mu_g\|^2 + \mathrm{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$.

- $(\mu_r, \Sigma_r)$: Real image feature의 평균과 공분산  
- $(\mu_g, \Sigma_g)$: Generated image feature의 평균과 공분산  

$\rightarrow$ 거리 기반 metric이므로 **0에 가까울수록** 좋음

<br>

## **3) Empirical 계산 과정**

### Step 1) Feature 추출
- Real image, Generated image 각각을 **InceptionV3의 pool3 layer**에 통과  
- 2048-dimensional feature vector 획득

<br>

### Step 2) 평균, 공분산 계산
Real set에 대해...

- $\mu_r = \frac{1}{N} \sum f_r$.
- $\Sigma_r = \frac{1}{N} \sum (f_r - \mu_r)(f_r - \mu_r)^T$.

<br>

Generated set에 대해 ...

- $\mu_g, \Sigma_g$ 도 동일하게 계산

<br>

### Step 3) Fréchet Distance 계산
두 정규분포 ($\mathcal{N}(\mu_r,\Sigma_r)), (\mathcal{N}(\mu_g,\Sigma_g)$) 사이의 거리  

<br>

## **4) Details**

### (a) 왜 FID가 IS보다 더 신뢰받는가?
- IS는 **real 데이터와 비교를 하지 않음**  
- FID는 **real distribution을 기준으로 생성품질을 직접 평가**  
- mode collapse를 잘 탐지함  
- 데이터 수가 늘어나면 더 안정적

<br>

### (b) Interpretation
**FID ↓ → Good**  

- 평균 차이가 작음  
- 분포 커버리지가 real과 유사  

<br>

**FID ↑ → Bad**  

- blur, artifact, 모드 collapse 등 모두 반영됨

<br>

## **5) Example (잘한/못한 케이스)**

### (a) Good

Real vs. Generated

- Real feature 분포: $(\mu_r = 2.0,\ \Sigma_r = 1.0)$.

- Generated feature 분포: $(\mu_g = 2.1,\ \Sigma_g = 1.05)$.


$FID = (2.0 - 2.1)^2 + (1.0 + 1.05 - 2\sqrt{1.0 \cdot 1.05})$.

- Mean difference term: $(0.1^2 = 0.01)$
- Covariance term:  $1.0 + 1.05 - 2\sqrt{1.05} \approx 2.05 - 2.049... \approx 0.001$.

<br>

$FID_{\text{good}} \approx 0.011$.

→ Real과 거의 동일한 분포 → **Excellent**

<br>

### (b) Bad

Real vs. Generated

- Real feature 분포: $(\mu_r = 2.0,\ \Sigma_r = 1.0)$.

- Generated feature 분포: $(\mu_g = 4.0,\ \Sigma_g = 3.0)$.

<br>

$FID = (2 - 4)^2 + (1 + 3 - 2\sqrt{3})$.

- Mean difference: $4$
- Covariance term:  $4 - 2\sqrt{3} \approx 4 - 3.464 \approx 0.536$.

<br>

$FID_{\text{bad}} \approx 4.536$.

→ 분포 차이가 매우 큼 → **Bad**

<br>

# 3. **KID (Kernel Inception Distance)**

- 생성 image와 실제 image의 분포 차이를 **MMD (Maximum Mean Discrepancy)** 기반으로 평가

- FID와 유사하게 feature space에서 분포를 비교
  
  $\rightarrow$ 차이점? **unbiased estimator**를 사용해 **small sample에서도 안정적**이라는 장점

<br>

## **1) 개념**

KID는  
- Real image 분포  
- Generated image 분포  

를 InceptionV3의 feature space에서 비교하고,  
**kernel-based MMD**를 사용하여 두 분포의 거리를 계산한다.

### 주요 특징
- FID와 달리 **공분산 행렬의 matrix square root 계산 필요 없음**  
- **unbiased estimator** → sample size 작아도 안정적  
- 값은 **작을수록 더 좋은 품질** (0에 가까울수록 Real과 유사)

<br>

## **2) 수식**

KID는 **polynomial kernel**을 사용한 MMD 거리:

\[
KID = \mathrm{MMD}^2(P_r, P_g)
\]

\[
\mathrm{MMD}^2 = \mathbb{E}[k(x_r, x_r')] + \mathbb{E}[k(x_g, x_g')] - 2\mathbb{E}[k(x_r, x_g)]
\]

여기서 kernel \(k\)는:

\[
k(x, y) = \left(\frac{1}{d} x^\top y + 1\right)^3
\]

- \(d\): feature dimension (InceptionV3 pool3 → 2048)
- \(x_r, x_r'\): real features
- \(x_g, x_g'\): generated features  

### Interpretation  
- Real–Real kernel similarity  
- Generated–Generated similarity  
- Real–Generated similarity  

을 비교하여 두 분포의 차이를 측정한다.

→ **KID 값이 낮을수록 real과 generated 분포가 유사함**

<br>

## **3) Empirical 계산 과정**

### Step 1) InceptionV3 Feature 추출
- Real images → \(f_r \in \mathbb{R}^{2048}\)  
- Generated images → \(f_g \in \mathbb{R}^{2048}\)

### Step 2) Kernel 계산
- 모든 pair에 대해 polynomial kernel 계산  
  \[
  k(f_i, f_j) = \left(\frac{1}{2048} f_i^\top f_j + 1\right)^3
  \]

### Step 3) Unbiased MMD estimator 계산
- Real–Real  
- Gen–Gen  
- Real–Gen  

pair들의 평균을 구해 MMD² 계산

### Step 4) Batch-averaging
- 흔히 mini-batch 단위로 여러 번 반복하여 평균  
  → variance 줄이기 위함

<br>

## **4) Details**

### KID vs FID
| 항목                | FID                                   | KID               |
| ------------------- | ------------------------------------- | ----------------- |
| 사용 모델           | InceptionV3                           | InceptionV3       |
| 비교 방식           | Fréchet distance(Gaussian assumption) | MMD(kernel based) |
| Estimator           | biased 가능                           | **unbiased**      |
| Small sample 안정성 | 낮음                                  | **높음**          |
| Computation         | matrix square root 필요               | 단순 kernel 연산  |

→ **소규모 데이터셋에서는 FID보다 KID가 더 신뢰적**

<br>

## **5) Example (잘한/못한 케이스)**

(이해를 위해 feature dimension을 단순화한 toy 예시)

---

### (a) Good Example

Real features:
- \(x_{r1} = 1.0,\ x_{r2} = 1.2\)

Generated features:
- \(x_{g1} = 1.1,\ x_{g2} = 1.0\)

**Step 1 — Kernel 계산**

Real–Real:
- \(k(1.0, 1.2) = (1 \cdot 1.2 + 1)^3 = (2.2)^3 = 10.648\)

Gen–Gen:
- \(k(1.1, 1.0) = (1.1 + 1)^3 = (2.1)^3 = 9.261\)

Real–Gen:
- \(k(1.0, 1.1) = (1.1 + 1)^3 = (2.1)^3 = 9.261\)

**Step 2 — KID 계산**

\[
KID_{\text{good}}
= 10.648 + 9.261 - 2(9.261)
\]

\[
= 10.648 - 9.261
= 1.387
\]

→ Real–Gen 차이가 작음 → **좋은 품질**

---

### (b) Bad Example

Real:
- \(x_{r1} = 1.0,\ x_{r2} = 1.2\)

Generated:
- \(x_{g1} = 3.0,\ x_{g2} = 3.2\)

Real–Real:
- 동일 → 10.648

Gen–Gen:
- \(k(3.0, 3.2) = (9.6 + 1)^3 = (10.6)^3 \approx 1191.016\)

Real–Gen:
- \(k(1.0, 3.0) = (3 + 1)^3 = 64\)

\[
KID_{\text{bad}}
= 10.648 + 1191.016 - 2(64)
\]

\[
= 1201.664 - 128
= 1073.664
\]

→ Real과 Generated의 similarity 차이가 매우 큼 → **Bad**

---

원하면 **FID/KID/IS 전체 비교표**, 또는 **PyTorch KID 구현 코드**도 만들어줄게!



# **FID∞ (Unbiased FID)**







### **1) 개념**





- FID는 샘플 수가 적으면 **편향(bias)** 발생
- 이 편향을 수학적으로 제거한 unbiased estimator
- 논문 Table 1에서 KL/RKL과 상관 가장 높은 메트릭 중 하나







### **2) 수식**





기본 FID 수식은 동일



- 단, \Sigma_r,\Sigma_g의 추정에 대해 unbiased correction 수행







### **3) empirical 계산**





- FID 계산 과정에 bias correction term 추가
- 여전히 샘플 mean/cov만 필요





------





# **🚀 7)** 

# **IS∞ (Unbiased IS)**







### **1) 개념**





- IS에서 KL 기대값의 finite-sample bias 제거
- 하지만 논문 결론: **IS의 근본적 문제는 해결 못함**







### **2) 수식**





IS_\infty = \exp\left( \mathbb{E}[ KL_\infty(p(y|x), p(y)) ] \right)





### **3) empirical 계산**





- Real/Gen 샘플 기반 p(y|x) 계산은 그대로
- KL term을 unbiased estimator로 대체





------





# **🚀 8)** 

# **Clean FID**







### **1) 개념**





- image 리사이징/aliasing 때문에 FID가 흔들리는 문제 해결
- 동일한 “clean” 전처리로 재계산하는 버전
- 논문에서 **가장 안정적(Inception 기반 중)**







### **2) 수식**





FID와 동일.





### **3) empirical 계산**





- 차이는 전처리:

  

  - anti-aliased resize
  - 일관된 interpolation

  

- 그 외는 FID 계산과 동일





------





# **🚀 9)** 

# **CLIP-FID (논문에서 제안)**







### **1) 개념**





- Inception feature를 CLIP feature로 대체

- 비-ImageNet 도메인에서 훨씬 Gaussian에 가깝고 안정적

  (논문 Table 2 및 Fig.6~7)







### **2) 수식**





FID_{CLIP} = FID(\mu_r^{CLIP},\Sigma_r^{CLIP},\mu_g^{CLIP},\Sigma_g^{CLIP})





### **3) empirical 계산**





- Inception 대신 CLIP에서 feature 추출
- mean·covariance 계산은 그대로





------





# **📌 마지막 초간단 요약**





각 메트릭은 “분포 기반”처럼 보이지만

**실제로는 모두 empirical sample(X_r, X_g) 기반 통계치로 근사하여 계산**한다.

| **Metric** | **개념**              | **필요한 정보**  | **실전 계산 방식**                |
| ---------- | --------------------- | ---------------- | --------------------------------- |
| KL / RKL   | 분포 비교             | likelihood       | 일반 모델 불가능, 논문에서만 가능 |
| IS         | 다양성 + 선명도       | p(y              | x)                                |
| FID        | Gaussian feature 비교 | feature mean/cov | 샘플 기반 통계 추정               |
| KID        | 비가우시안 MMD²       | kernel           | feature 쌍 kernel 평균            |
| FID∞ / IS∞ | 편향 제거             | feature mean/cov | unbiased 통계 추정                |
| Clean FID  | 전처리 통일           | feature mean/cov | anti-aliased resize 후 FID        |
| CLIP-FID   | CLIP feature 기반 FID | CLIP feature     | feature mean/cov                  |



------



필요하면 **각 메트릭의 장단점만 따로 정리**해줄 수도 있어!