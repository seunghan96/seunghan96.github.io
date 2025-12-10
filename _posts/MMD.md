# Maximum Mean Discrepancy (MMD)

<br>

# 1. **개념 / 직관**

## **1) 두 분포가 같은지(동일한 분포인지) 비교하는 통계적 거리**

MMD는 다음 질문에 답하기 위한 거리 지표이다:

> **“Real 분포 P와 Generated 분포 Q가 같은 분포인가?”**



이를 sample만 가지고 판단할 때 사용하는 **non-parametric two-sample test**이다.

즉, 정규분포 가정이나 모수적 모델 없이 **임의의 분포**도 비교할 수 있다.



------





## **2) Feature space에서 평균(Mean)의 차이를 비교**





MMD의 핵심 아이디어는 다음:



- 데이터를 kernel function으로 **고차원 feature space**로 보낸다 (Feature map: φ(x)).

- 그 공간에서의 평균을 계산한다.

  

  - Real 분포 → 평균 μ_P
  - Generated 분포 → 평균 μ_Q

  





**MMD = 두 평균의 거리**



→ 분포가 같다면 두 평균도 같다

→ 분포가 다르면 평균이 달라지고, 거리가 커진다



------





## **3) Kernel trick**

## **으로 실제 feature map 없이 계산**





φ(x)로 mapping하지 않더라도,

**kernel k(x, y) = φ(x)ᵀ φ(y)** 를 이용하여

계산을 효율적으로 수행할 수 있다.



------





## **4) 직관적 요약**





- 두 분포에서 샘플을 몇 개 가져와서
- Real–Real similarity, Gen–Gen similarity, Real–Gen similarity를 비교하여
- **Real–Gen가 더 다르면 → MMD 커짐**
- **Real과 Gen가 비슷하면 → MMD 작아짐 (0에 가까움)**





------





# **2.** 

# **수식**







## **1) Population version (이론적 정의)**





\mathrm{MMD}^2(P, Q) = \|\mu_P - \mu_Q\|_{\mathcal{H}}^2



이를 kernel로 풀면:



\mathrm{MMD}^2(P, Q) = \mathbb{E}_{x,x' \sim P}[k(x,x')] + \mathbb{E}_{y,y' \sim Q}[k(y,y')] - 2\mathbb{E}_{x \sim P, y \sim Q}[k(x,y)]



------





## **2) Unbiased empirical estimator**





Real sample: \{x_1, x_2, \dots, x_m\}

Gen sample: \{y_1, y_2, \dots, y_n\}



\widehat{\mathrm{MMD}}^2 = \frac{1}{m(m-1)} \sum_{i \neq j} k(x_i, x_j) + \frac{1}{n(n-1)} \sum_{i \neq j} k(y_i, y_j) - \frac{2}{mn} \sum_{i,j} k(x_i, y_j)



- 첫 번째: Real–Real similarity 평균
- 두 번째: Gen–Gen similarity 평균
- 세 번째: Real–Gen similarity 평균





→ Real–Gen similarity가 매우 낮으면 MMD 값 증가



------





## **3) Kernel 예시**





가장 많이 사용하는 kernel들:



------





### **(a) Gaussian (RBF) kernel)**





k(x, y) = \exp\left( - \frac{\|x-y\|^2}{2\sigma^2} \right)



------





### **(b) Polynomial kernel (KID에서 사용)**





k(x, y) = \left( \frac{1}{d} x^\top y + 1 \right)^3



------





# **3.** 

# **예시 (직관적인 toy example)**





Real sample:

x_1 = 1,\quad x_2 = 1.2



Generated sample:

y_1 = 3,\quad y_2 = 3.2



Kernel: **Linear kernel** k(a,b) = ab

(단순화를 위해 선택함)



------





## **Step 1 — Real–Real similarity**





k(1, 1.2) = 1 \cdot 1.2 = 1.2



Real–Real 평균 = 1.2



------





## **Step 2 — Gen–Gen similarity**





k(3, 3.2) = 9.6



Gen–Gen 평균 = 9.6



------





## **Step 3 — Real–Gen similarity**





k(1,3) = 3,\quad k(1,3.2)=3.2



평균 = (3 + 3.2) / 2 = 3.1



------





## **Step 4 — MMD 계산**





\mathrm{MMD}^2 = 1.2 \;+\; 9.6 \;-\; 2(3.1)



= 10.8 - 6.2 = 4.6



→ Real과 Generated의 차이가 크므로 **MMD가 큼**



------





## **Good example 비교**





Generated sample이 Real과 비슷하다고 가정:



Gen sample:

y_1 = 1,\quad y_2 = 1.1





### **계산:**





Real–Real similarity:

k(1,1.2) = 1.2



Gen–Gen similarity:

k(1,1.1)=1.1



Real–Gen similarity:

k(1,1)=1,\quad k(1,1.1)=1.1

평균 = 1.05



\mathrm{MMD}^2 = 1.2 + 1.1 - 2(1.05) = 2.3 - 2.1 = 0.2



→ Real과 매우 비슷 → **MMD 작음 (좋다)**



------





# **요약**



| **지표**                              | **의미**                        |
| ------------------------------------- | ------------------------------- |
| **MMD = 두 분포의 평균임베딩 차이**   | Real/Gen similarity 자체를 비교 |
| **kernel 기반**                       | Gaussian, Polynomial 등         |
| **Unbiased estimator 가능**           | 작은 샘플에서도 안정적          |
| **GAN 평가(KID), 통계적 검정에 활용** | 실전에서 매우 자주 등장         |



------



필요하면



- **RBF kernel 기반 MMD PyTorch 코드**,

- **KID와 MMD 차이 표**,

- **GAN 학습에서 MMD loss 사용하는 법**

  도 만들어줄게!