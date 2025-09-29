# Rotary Positional Embedding (RoPE)

최근 Transformer 계열 모델(특히 GPT, LLaMA, ChatGLM 등)에서 많이 쓰이는 PE 기법

------





# 1. Positional Embedding (PE)

Transformer는 order-invariant

$\rightarrow$ Positional Embedding (PE)가 필요

<br>

Previous works

- **Absolute** PE: 각 위치에 고정된 vector를 더함 (예: BERT) → 길이 고정, extrapolation 불가.
- **Relative** PE: 위치 간 상대적 거리만 반영 (Transformer-XL, T5 등).

<br>

# 2. Relative PE의 등장

**Absolute Positional Embedding의 문제점**

1. **길이 일반화 불가 (extrapolation 문제)**

   - "절대" 위치마다 "고정된" vector를 더하기 떄문에!
   - (학습 시) 본 적 없는 **더 긴 sequence 길이**에서는 불가능

2. **상대적 거리 (relative distance) 정보 부족**

   - 절대 위치만 더해주므로, ***“두 token이 얼마나 떨어져 있는지”***를 직접적으로 알기 어려움.

3. **비효율적 표현**

   - 같은 상대 거리 (relative distance), 하지만 absolute embedding은 서로 다른 vector를 가짐 → redundancy

   - e.g.,

     -  position 5와 6
     - position 100과 101

     $\rightarrow$ 모두 상대적 거리가 1인데, absolute embedding은 완전히 다른 vector

<br>

**Relative Positional Embedding의 등장**

- 토큰 쌍 $(i,j)$에 대해 **"상대적 거리 $i-j$**"를 embedding
  - 즉, (token 개수가 $N$개 라면) $N\times N$ 짜리 attention matrix에 더하게 됨.
- 장점:
  - **길이 일반화** 가능 → 학습 길이보다 긴 문장에도 잘 적용.
  - **상대 거리 정보**가 직접 attention에 들어감 → 문법/구조 학습에 유리.
  - **효율성** → 같은 상대 거리에 같은 embedding 사용.

<br>

# 3. Relative PE의 한계점: 느림

Absolute PE

- 길이 $N$인 sequence에 대해 **PE** $N$**개**만 준비 → 각 토큰 임베딩에 단순히 더해줌!

  → attention 연산( $QK^\top$ )에는 **추가 연산 없음**.

- 추가 비용: 선형($O(N·d)$)

<br>

Relative PE

- Attention score를 계산할 때:
  - $\text{Score}(i,j) = \frac{q_i^\top k_j}{\sqrt{d}} + b_{i-j}$
- 여기서 $b_{i-j}$는 **"두 위치 간 상대 거리"**를 lookup해서 가져오는 값

$\rightarrow$ 즉, 모든 $(i,j)$ 쌍에 대해 한 번씩, $N \times N$ 테이블 수준으로 bias를 추가해야 함.

<br>

Summary

- **Absolute**: Token 개수 $N$만큼만 추가 연산 

  → attention 계산은 깔끔한 GEMM (행렬곱) 하나.

- **Relative**: attention 행렬 자체 ($N \times N$)에 element-wise 연산이 들어감.

  - N이 크면 $(N^2)$ 항의 overhead가 커짐.

<br>

# 4. Key Idea of RoPE

**RoPE**의 두 가지 핵심

- **relative 위치를 “내적 구조”**에 직접 녹여냄
- 단순하면서도 **길이 확장성**이 뛰어남

<br>

### Details

각 쿼리(Q), 키(K) vector에 **"위치에 따라 회전(Rotation)을 적용"**

$\rightarrow$ 이렇게 하면 **내적(Q·K)** 계산 시, 두 위치 간 **상대적 거리 정보**가 자연스럽게 포함

<br>

즉, ***“어디에 있는지(absolute)”***가 아니라 ***“얼마나 떨어져 있는지(relative)”***가 attention score에 반영

<br>

# 5. Rotation Matrix

## (1) 회전이란

2차원 평면에서 어떤 점 $(x, y)$를 ***"원점"을 기준***으로 돌린다고 해보면

- e.g., (1, 0)을 90°(시계 반대) 돌리면 → (0, 1)

<br>

“점들을 일정한 각도만큼 돌린다” = **회전 변환**

<br>

## (2) 회전 "행렬"

선형대수에서는 이런 변환을 **행렬 곱**으로 표현 가능!

- 점: 열벡터 $\begin{bmatrix}x \\ y\end{bmatrix}$
- 변환: 행렬 $R$
- 변환된 점: $R \begin{bmatrix}x \\ y\end{bmatrix}$

<br>

## (3) 2D 회전 행렬 공식

원점을 기준으로 $θ$만큼 시계 반대 방향 회전시키는 행렬:

- $R(\theta) = \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix}$.

<br>

$\begin{bmatrix} x’ \\ y’ \end{bmatrix} \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$ $= \begin{bmatrix} x\cos\theta - y\sin\theta \\ x\sin\theta + y\cos\theta \end{bmatrix}$.

<br>

## (4) 직관

(1,0)을 90° 돌린다:

$R(90^\circ)\begin{bmatrix}1\\0\end{bmatrix} = \begin{bmatrix}0\\1\end{bmatrix}$.

→ 오른쪽을 보던 화살표가 위쪽으로

<br>

(0,1)을 90° 돌린다:

$R(90^\circ)\begin{bmatrix}0\\1\end{bmatrix} = \begin{bmatrix}-1\\0\end{bmatrix}$.

→ 위쪽을 보던 화살표가 왼쪽으로

<br>

## (5) 성질

1. **길이 보존**: 
   - 회전은 “크기”를 바꾸지 않음
   - 단순히 방향만 바꿈!!
   - $|(x, y)| = |(x’, y’)|$.
2. **각도 보존**: 
   - 두 벡터 사이 각도도 변하지 않음.
3. **직교행렬**: 
   - $R(\theta)^\top R(\theta) = I$. (전치 곱하면 항등행렬).
4. **determinant = 1**: 
   - 순수 회전(크기 변화나 반사 없음)

<br>

## (6) Summary

- 회전 행렬 = “점(벡터)을 일정한 각도만큼 도는 공식”을 행렬로!

- (공식) $R(\theta) = \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix}$

- 크기/각도 보존 & 단순히 방향만 바꿈!!
- (RoPE에서는) 이 회전 성질을 이용해 **위치 차이(=상대적 거리)**를 자연스럽게 encoding!

<br>

# 6. RoPE

## (1) 수식

입력: $x \in \mathbb{R}^d$ 

위치: $m \in \mathbb{Z}^+$.

RoPE 적용: $\text{RoPE}(x, m) = R_m \, x$.

- $R_m$: **2D rotation matrix**가 block-diagonal 형태로 d/2개 반복된 것.

<br>

즉, 차원을 짝수로 나눠서:

$R_m = \begin{bmatrix} \cos(m\theta_1) & -\sin(m\theta_1) & & & \\ \sin(m\theta_1) &  \cos(m\theta_1) & & & \\ & & \cos(m\theta_2) & -\sin(m\theta_2) & \\ & & \sin(m\theta_2) &  \cos(m\theta_2) & \\ & & & & \ddots \end{bmatrix}$.

<br>

## **(2) 주파수 정의**

각각의 $\theta_i$는 고유 주파수:

- $\theta_i = 10000^{-2(i-1)/d}$.

이는 기존 sinusoidal embedding과 동일한 스케일링.

<br>

## **(3) Q, K에 적용**

Q, K에 동일한 RoPE 적용:

- $Q’_m = R_m Q$.
- $K’_n = R_n K$.

<br>

Attention score:

- $(Q’_m)^\top K’_n = Q^\top (R_m^\top R_n) K$.

<br>

여기서 $R_m^\top R_n = R_{n-m}$ 이므로, **점수는 m,n의 차이(=상대 위치)**에만 의존합니다.

<br>

## (4) 해석

- RoPE는 사실상 각 위치별로 **vector를 특정 각도로 회전**시킵니다.
- 두 벡터 사이의 상대적 위치 차이가 반영됨.
- (절대 위치가 아니라) 상대 위치로 바뀌므로 **sequence 길이가 늘어나도 동작**.

<br>

## (5) 긴 sequence를 핸들링하는 이유

- 절대 위치 vector를 더하는 방식: (학습한 길이 이상의) **extrapolation 불가**!

- RoPE는 **회전 각도 공식이 연속적/주기적**

  $\rightarrow$ 원하는 만큼 큰 $m$에 대해서도 $R_m$을 계산할 수 있음.

- 즉:

  - 길이가 늘어나도 위치 index $m$만 커지면 됨.
  - 기존 학습 범위를 넘어선 길이도 generalize 가능.

- 다만, 각도 주파수는 주기성을 갖기 때문에 **아주 길어지면 aliasing/주기 오버랩 문제**가 발생할 수 있어,

  이를 보완하기 위해 **RoPE scaling (NTK-aware scaling, Linear scaling 등)** 기법이 연구되어 적용됩니다 (LLaMA-2/3).





------





# **6. 장점 요약**





✅ 상대적 위치 정보 반영 (자연스러운 sequence 처리)

✅ 학습 길이 외에도 extrapolation 가능

✅ 구조 단순, 구현 쉬움 (회전행렬 적용)

✅ QK 내적에 직접 반영되어 효율적



------





# **7. PyTorch 의사코드**



```
import torch

def apply_rope(x, position_ids, dim):
    """
    x: (batch, seq_len, dim)
    position_ids: (seq_len,) 각 토큰 위치
    dim: 전체 차원 (짝수)
    """
    half_dim = dim // 2
    freq_seq = torch.arange(half_dim, dtype=torch.float32)
    freq = 10000 ** (-2 * freq_seq / dim)   # (half_dim,)

    # 각 위치의 각도
    theta = torch.einsum("i,j->ij", position_ids.float(), freq)  # (seq_len, half_dim)

    cos, sin = torch.cos(theta), torch.sin(theta)
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    x_rot = torch.cat([x1 * cos - x2 * sin,
                       x1 * sin + x2 * cos], dim=-1)
    return x_rot
```



------





# **✅ 정리**





- **RoPE = Rotary Positional Embedding**
- 각 위치에 따라 vector를 **회전**시키는 방식으로 위치 정보를 QK 내적에 주입
- Attention 점수가 **상대적 위치 차이**에만 의존 → 길이 늘려도 동작
- 매우 긴 sequence에서는 scaling 기법과 함께 사용





아주 좋은 질문이에요 👍

겉으로 보면 RoPE도 **“두 위치 차이 i-j”**가 attention score에 반영되니까,

“그럼 Relative PE처럼 N\times N 항을 다 만들어야 하는 거 아닌가?” 하는 의문이 들 수 있습니다.



하지만 **핵심 차이**는 “**어느 시점에 상대 위치 정보를 반영하느냐**”예요.



------





## **🔹 Relative PE**





- Attention score 계산식에 **추가 항**으로 직접 들어감:

  \text{Score}(i,j) = \frac{q_i^\top k_j}{\sqrt{d}} + b_{i-j}

- 즉, (i,j) 쌍마다 **별도의 bias**를 더해줘야 함.

- 이 추가 연산이 **attention 행렬 전체 N \times N**에서 필요 → 메모리/연산 오버헤드↑.





------





## **🔹 RoPE**





- Q, K 벡터에 **사전에 회전 변환** 적용:

  q_i’ = R_i q_i,\quad k_j’ = R_j k_j

- 그러면 attention score는:

  (q_i’)^\top k_j’ = q_i^\top (R_i^\top R_j) k_j = q_i^\top R_{j-i} k_j

- 수학적으로는 여전히 “상대 거리 j-i” 효과가 들어가지만,

  **실제로는 토큰 레벨에서 Q,K만 변환**해 두면 됨.





즉, 각 위치 i에 대해 R_i**만 적용**하면 끝 → O(Nd) 비용.

이후 attention은 그냥 기존대로 Q’K’^\top 한 번의 행렬곱으로 계산.



------





## **⚡️ 차이의 본질**





- **Relative PE**: “attention score 행렬 만든 뒤”에 위치 정보 더함 → N^2 위치쌍 처리 필요.
- **RoPE**: “attention 들어가기 전에 Q,K에 위치 정보를 녹임” → 이후는 그냥 matmul.





------





## **✅ 정리**





> RoPE도 수학적으로는 **상대 거리 정보**를 반영하지만,

> 구현은 **Q,K 벡터 선처리만 하면 되고 attention score 계산은 그대로**라서

> **NxN 오버헤드가 추가되지 않습니다.**

> 이게 RoPE가 효율적이고 빠른 이유예요 🚀.



------



👉 혹시 제가 그림으로 **Relative PE vs RoPE 데이터 흐름(토큰 단계 vs 어텐션 행렬 단계)**를 비교해서 보여드릴까요?