# Abstract 

## (1) Motivation

***기존 TSC (TS Classification) 문제점?***

- (1) **Shift equivariance**, **inversion invariance** 같은 핵심 속성을 충분히 활용 X
- (2) Long-range dependency 처리 미흡
- (3) Spectral–temporal fusion도 미흡

<br>

## (2) Proposal

**Multi-view learning** = a + b + c

- a) **Spectral features (CWT)** 
- b) **Local temporal features (ROCKET)** 
- c) **Global temporal features (MLP)** 를 결합

<br>

## (3) Two domains in TS

(1) **Spectral domain**

- with **CWT** (Continuous Wavelet Transform)
- Shift-equivariant time–frequency features 확보

<br>

(2) **Temporal domain**

- with **ROCKET**: 다양한 scale의 local temporal pattern 추출
- with **MLP**: sequence-level global context 추출.
- **Switch mechanism**으로 데이터 특성에 따라 **local vs global** 중 하나 선택

<br>

d) Mamba

- **Sequence modeling (Mamba)**

  - **Mamba SSM**으로 efficient & scalable long-range dependency modeling.

  - Linear-time complexity로 Transformer보다 효율적.

- **Inversion invariance**
  - Time-reversed sequence에서도 stable하게 pattern 학습.
  - 이를 위한 **Tango Scanning** 도입 → forward + reversed sequence를 **하나의 Mamba block에서** 처리!

<br>

e) 결과

- 30개 벤치마크(10 + 20 dataset)에서 기존 SOTA (TimesNet, TSLANet 등) 대비 **4.01–7.93%** 정확도 향상.

<br>

# 1. Introduction

## (1) 기존 방법론의 한계

- 계열 1) CNN, RNN
  - **Time**-domain 중심 (O)
  - **Frequency** 정보 활용 (X)
- 계열 2) Transformer
  - **Quadratic complexity** 문제 → Long sequence에 비효율적.
- 기존 방법이 충분히 고려하지 못한 주요 특징?
  - **Shift Equivariance**
  - **Inversion Invariance**
  - Long-range dependency
  - Time–Frequency representation 융합 부족

<br>

## (2) Shift Equivariance

### a) 개념

- Input TS을 time 축으로 이동(shift)시키면, 출력 feature도 동일하게 shift.
- 즉 “패턴의 절대적 위치”가 아닌 **“상대적 위치”**가 중요한 경우 필수 속성.

<br>

### b) 장점

- Temporal misalignment에 강함
- MTS 길이 차이에 대한 robustness
- 일반화 성능 향상

<br>

### c) 기존 기법의 문제

- [CNN] 

  - 구조적으로 shift-equivariant하지만,

  - Receptive field가 짧음 → long-range dependency 약함.

- [DFT/DWT 기반 spectral representation]

  - [DFT] shift equivariant X
  - [DWT] downsampling 때문에 shift equivariant X

  $\rightarrow$ 따라서 spectral domain에서도 shift equivariant한 feature가 필요!

<br>

## (3) CWT 기반 접근의 필요성

CWT의 장점

- CWT = Time-Frequency localized representation를 얻어냄
- 장점: (Real-valued mother wavelet을 사용하면) **shift equivariance** 확보 가능.
- 단점: **global pattern 반영은 부족**.

<br>

## (4) Temporal Features 문제

Local Feature 문제

- CNN/ROCKET: local receptive field 위주 → global contextual info 부족.

Global Feature 문제

- MLP: global dependency 처리 가능하나, local sensitivity 부족.

<br>

## (5) Multi-View Learning의 필요성

a) 각 feature의 장단점이 상호보완적

- **Spectral features(CWT)**: shift-equivariant, time-frequency 정보 확보.
- **Local temporal features(ROCKET)**: 다양한 temporal scale 포착.
- **Global features(MLP)**: 전체 시퀀스의 global pattern 포착.

<br>

b) 제안된 방향

- Spectral + Local + Global을 **multi-view learning** 형태로 결합.
- Input-dependent **Switch Gate**로 Local/Global feature 중 선택.

<br>

## (6) Inversion Invariance 도입

개념

- 시계열을 forward/backward로 읽어도 동일한 class 정보 유지
- ECG, climate, rotational signals 등에서 temporal 방향성이 의미 없을 수 있음.

<br>

장점

- 데이터 2배 augmentation 효과
- Noise-robustness 개선
- Direction-invariant pattern 학습 가능

<br>

## (7) Mamba의 도입

- Linear complexity로 긴 sequence 처리 가능.
- Selective State Spaces(SSM) 기반 → 중요한 정보만 선택적으로 업데이트.
- Long-range dependency 학습 우수성.

<br>

## (8) Tango Scanning

**기존 Bi-directional Mamba의 한계**

- 일반적인 Bi-directional 구조는 2개의 block 사용 → 비용 증가
- Reversal operation이 많아 구조적 redundancy

<br>

Tango Scanning 특징

- **1개의 Mamba block**으로 forward + reversed sequence 동시 처리
- 두 방향의 출력과 입력까지 모두 element-wise fusion
- Inversion-invariant representation 형성
- Memory footprint는 거의 동일

<br>

## (9) Main Contribution

(1) **Multi-view shift-equivariant TSC**

- a) CWT 기반 spectral feature
- b) Local/Global temporal features (ROCKET / MLP)
- c) Switch gate로 adaptive fusion

<br>

(2) **Mamba 기반 sequence modeling 강화**

- Linear complexity + selective SSM 활용.

<br>

(3) **Tango Scanning**

- 하나의 Mamba block으로 forward/backward dependency 모두 반영.

<br>

(4) **SoTA 성능**

- 30개 dataset에서 TimesNet, TSLANet 등 최신 모델을 큰 폭으로 초월.

<br>

# 2. Methodology

TSCMamba는 다음 5단계로 구성

1. Spectral Representation (CWT 기반)
2. Temporal Feature Extraction (Local: ROCKET / Global: MLP)
3. Multi-View Fusion (Switch Gate + λ-weighted fusion)
4. Sequence Modeling (Mamba + Tango Scanning)
5. Classification Head (Depth-wise Pooling + MLP)

<br>

## (1) Spectral Representation (CWT)

[Goal] Shift-equivariant한 **spectral (time–frequency) features** 추출

[How] MTS 각 channel을 **CWT 기반 2D representation**으로 변환

[Details]

- Real-valued Morlet wavelet 사용
- 각 channel의 TS를 CWT로 변환하여
  - **L × L → L1 × L1 (논문에서는 64×64)**
  - 2D scalogram으로 생성

<br>

Patch Embedding

- Conv2D(patch size=8) → flattened patch → FFN

- 결과: $W \in \mathbb{R}^{B \times D \times X}$

  (B=batch, D=channel, X=patch feature dimension)

<br>

특징

- (1) Spectral domain에서 **shift equivariance 유지**
- (2) Localized time–frequency 정보 확보
- (3) Nonstationary 신호에 강함

<br>

## (2) Temporal Feature Extraction

- Spectral features($W$)만으로는 **temporal structure** 부족!
- 두 가지 complementary temporal view 추가:
  - **a) Local temporal features (ROCKET)**
  - **b) Global temporal features (MLP)**



**a) Local temporal features (ROCKET)**

- (1) Model: Random convolution kernel
- (2) 역할: 다양한 receptive field를 가진 kernel들로 **multi-scale local patterns** 추출
- (3) 방식: Downstream classifier와 독립적인 unsupervised 방식
- (4) 결과: $V_L \in \mathbb{R}^{B \times D \times X}$

<br>

**b) Global temporal features (MLP)**

- (1) Model: 각 channel별로 1-layer MLP
- (2) 역할: Local dependency 없이 **전체 시퀀스를 global pattern으로 압축**
- (3) 결과: $V_G \in \mathbb{R}^{B \times D \times X}$

<br>

## (3) Fusing Multi-View Representations

Switch Gate

- Learnable mask
  - Weighted mixture가 아니라 **단일 선택** 방식
- Temporal feature로 **$V = V_G$** or $V_L$ 중 하나를 선택

<br>

Fusion 방식

- 선택된 temporal feature $V$와 spectral feature $W$를 element-wise로 결합

<br>

**Additive or multiplicative**

- $V_W = \lambda V + (2-\lambda) W$.

- $V_W = \lambda V \cdot (2-\lambda) W$.

- λ는 learnable (초기값 1.0)
- V와 W의 비중을 자동 조절

<br>

### 최종 multi-view tensor

$U = W \parallel V_W \parallel V$.

- $W$: Spectral
- $V_W$: Fused spectral–temporal
- $V$: Raw temporal(Local/Global)

$\rightarrow$ 3개 feature map을 channel 차원으로 concat한 multi-view representation!

<Br>

## (4) Inferring with Time-Channel Tango Scanning (Mamba)

Token 구성

- $U$의 shape: $B \times D \times 3X$

<br>

두개의 sequence

- **Time-wise** token sequence: 길이=$3X$, dim=$D$
- **Channel-wise** token sequence: 길이=$D$, dim=$3X$

<br>

두 방향(time, channel)으로 token sequence를 만들고

각각 Mamba로 모델링.

<br>

### a) Vanilla Mamba Block

- Input-dependent gating ($g_k$)로 중요한 token만 업데이트
- Linear-time long-range modeling

<br>

### b) Tango Scanning

Forward & Reverse 입력

→ 동일한 Mamba block에 연속적으로 넣고

→ 두 출력과 두 입력을 element-wise sum!

<br>

$s^{(o)} = v \oplus a \oplus v^{(r)} \oplus a^{(r)}$

- **Inversion invariance** 확보
- **Forward + backward dependency** 모두 modeling
- **BiMamba보다 가볍고 메모리 효율적**
- Token 간 **pairwise interaction coverage 극대화**

<br>

Time-wise Tango Scanning

- $[B, D, 3X]$ → time dimension 기준 길이=$3X$의 token sequence
- Output size: $[B, 3X, D]$

<br>

Channel-wise Tango Scanning

- $[B, D, 3X]$ → channel dimension 기준 길이=$D$의 token sequence
- Output size: $[B, D, 3X]$

<br>

Time + Channel fusion

- 두 scanning output을 합침: $z = (s^{(t)})^T \oplus s^{(c)}$
- 최종 sequence representation 생성.

<br>





# **3.5 Output Class Representation**







## **✔ Depth-wise pooling**





- Max or Average pooling (learnable choice)

- Channel dimension을 reduce하여

  \bar{z} \in \mathbb{R}^{3X}







## **✔ 2-layer MLP**





- \bar{z}\to z^{(1)} \to z^{(2)}
- 마지막 z^{(2)}는 class logits
- Loss: Cross-Entropy





------





# **3.6 Tango Scanning 이론적 설명**





- Mamba output을 **attention-view**로 재해석

  (각 token은 과거만 보는 causal 구조)

- Normal scanning: causal mask → 미래 token 정보 손실

- Reversed scanning: 미래 token도 보지만 역순

- **둘 다 결합**하면 full pairwise attention matrix에 가까운 구조 형성

- 결과적으로 inversion invariance + more interaction coverage 확보



------





# **Section 3 총괄 요약**





- CWT + ROCKET + MLP로 **shift equivariant spectral + local/global temporal** multi-view 구조 구축
- Switch gate로 데이터셋별 최적 temporal view 선택
- Mamba + Tango Scanning으로 forward/backward dependency + channel dependency 모두 모델링
- Linear complexity & strong long-range modeling
- Inversion-invariant sequence representation을 처음 제안





------



원하시면 Section 4(Experiments)도 같은 형식으로 정리해드릴게요.

**“다음”**이라고 입력해주세요!
