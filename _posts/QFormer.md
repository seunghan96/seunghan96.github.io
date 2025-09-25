# 1. Q-Former

(BLIP-2: https://arxiv.org/pdf/2301.12597)

<br>

## (1) 목적

- **(Frozen) Vision Encoder** & **(Frozen) LLM** 사이의 **"adapter"**.

- 소수의 **"learnable 질의 토큰 (query tokens)"** 이 Vision 특징을 **선택적으로 추출**

  $\rightarrow$ LLM이 사용하기 좋게 **"짧은 token sequence"**로 정렬

<br>

## (2) 구조

Notation

- Patch embedding: $V\in\mathbb{R}^{N\times d_v}$
  - Vision encoder를 통과해서 나옴
- Learnable query tokens: $Q\in\mathbb{R}^{M\times d}$ (보통 수십 개)

<br>

Q-Former의 구조

- (1) **Self-Attn**
- (2) **Cross-Attn(Q↔V)**
- (3) **FFN** 

블록을 쌓은 **소형 Transformer**

$\rightarrow$ 산출된 $M$개 token을 **projector**로 LLM 임베딩 차원에 맞춰 투사함

<br>

Cross-Attention

- $\text{Attn}(Q,K,V)=\text{softmax}\Big(\frac{QW_Q (KW_K)^\top}{\sqrt{d}}\Big)\, VW_V$.
  - K,V: Patch embedding
  - Q: Query tokens
  - 위 결과로 나온 token을 LLM 입력으로 사용

<br>

## (3) Pretraining tasks

- **ITC** (image-text contrastive)
- **ITM**(image-text matching)
- **이미지 조건 LM**(캡셔닝) 

<br>

# 2. 비디오용 Q-Former

비디오는 **시간축**이 생김!

$\rightarrow$  Q-Former를 **frame × 공간 × 시간**으로 확장!

<br>

## (계열 1) Frame-wise → Cascaded Q-Former

예시: **Video-LLaMA**

<br>

Procedures

- 단계 1: **Frame-wise**

  - Frame 별 token 생성
  - How? 각 frame 특징에 **공유 가중치 Q-Former**를 적용

- 단계 2: Temporal aggregation

  - **두 번째 Q-Former**가 frame들 간 **시간 통합** (Temporal aggregation)을 수행

- 추가: **Frame 임베딩 & 시간 Positional Encoding** 주입, **오디오용 Audio Q-Former** 병행 가능.

  → 장점: 이미지 Q-Former 설계를 **직관적으로 비디오로 확장**, 모듈성 높음.

  → 단점: **두 단계**로 인해 토큰/계산 예산이 늘고, 시간 관계를 명시적으로 강하게 모델링하진 못할 수 있음. 

<br>

## (계열 2) **Spatio-Temporal Q-Former (단일기)**

예시: **Video Q-Former**

- **하나의 Q-Former**가 **공간 & 시간을 동시에 질의**하도록 설계

- 입력 전에 **Attentive pooling 모듈**로 frame/구간 중요도를 가중.

- 내부에 **Expert (e.g., SP-FFN, T-FFN, SM-FFN)** 를 두어 **공간/시간/의미 정렬**을 분담.

  → 장점: **시간 의존성**을 Q-Former 내부에서 직접 모델링, 장기 문맥에 유리.

  → 단점: 구조가 다소 복잡, 구현 난이도↑. 

<br>

## **수식/목표의 차이(요약)**

- **수식 관점**: 
  - 이미지형의 $Q$가 frame 단위 특징 $V_t$만 보던 것에서, 
  - 비디오형은 $\{V_{t}\}_{t=1..T}$ 전체에 대해 **시간 인지형 Cross-Attn**을 구성
    - 시간 PE/frame 임베딩 포함
- Cascaded: 
  - $\text{QF}^{(1)}(V_t)\to Z_t$
  - $\text{QF}^{(2)}([Z_1,\dots,Z_T])$
- Spatio-temporal: 한 번의 $\text{QF}_{\text{S$\rightarrow$ T}}(\{V_t\})$로 통합.

<br>

# 3. Pseudocode

### (1) **이미지 Q-Former (BLIP-2)**

```python
V = img_encoder(img)          # [N, d_v], frozen
Q = learnable_queries(M, d)   # [M, d]
Z = QFormer_cross_attn(Q, V)  # [M, d]

tokens = proj_to_llm(Z)       # -> feed to LLM
```

<br>

### (2) **비디오 Cascaded Q-Former (Video-LLaMA식)**

```python
V_t = [img_encoder(frame_t) for t in 1..T]
Z_t = [QFormer1(Q, V_t[t] + time_PE[t]) for t in 1..T]  # frame-wise
Z   = QFormer2(Q, concat(Zt))                           # temporal integrate

tokens = proj_to_llm(Z)
```

<br>

### (3) **비디오 Spatio-Temporal Q-Former (단일기)**

```python
V = [img_encoder(frame_t) for t in 1..T]
V_hat = attentive_pool(V)            # 중요 구간 가중
Z = QFormer_ST(Q, V_hat, time_PE)    # 한 번에 공간+시간 질의

tokens = proj_to_llm(Z)
```

<br>

# 4. Summary

| **항목**    | **이미지 Q-Former** | **비디오 Q-Former (Cascaded)**                  | **비디오 Q-Former (Spatio-Temporal)** |
| ----------- | ------------------- | ----------------------------------------------- | ------------------------------------- |
| 질의 범위   | 공간                | 공간 -> 시간                                    | 공간 & 시간                           |
| 토큰 예산   | M (소수)            | 1단계 (공간/frame별) + 2단계 (통합)로 다소 증가 | 단일 모듈, 설계에 따라 효율적         |
| 시간 인코딩 | 필요 없음           | frame 임베딩/시간 PE                            | 시간 PE + 내부 시간 상호작용          |
| 대표 예     | BLIP-2              | Video-LLaMA                                     | Video Q-Former (ICLR’25 under review) |
| 대표 목표   | ITC/ITM/캡션        | Video-to-Text, VTM/VTC/VTG                      | VTM/VTC/VTG + 주의적 풀링             |
