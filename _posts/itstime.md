# Abstract

**문제 제기**

- 기존 TSFM benchmark의 한계점:
  - (1) **Legacy dataset 재사용** 중심의 data 구성
  - (2) **Data integrity** 부족
  - (3) **Real-world와 맞지 않는** task formulation
  - (4) **Dataset-level** evaluation에 머무는 분석 관점

<br>

**TIME Benchmark 제안**

- **50개** fresh dataset, **98개** forecasting task
- strict **zero-shot** evaluation 설계
- **data leakage 방지**

<br>

**Benchmark Construction**

- **LLM** + **human expert** 기반 human-in-the-loop pipeline
- **고품질 data** 정제 및 검증
- real-world operational requirement와 variate predictability에 aligned된 **task-centric 설계**

<br>

**Evaluation Perspective**

- Dataset-level → **Pattern-level** evaluation로 확장
- **Structural** time series feature 기반 분석
- Temporal pattern 중심의 generalizable insight 제공

<br>

**Empirical Study**

- 12개 TSFM evaluation
- Multi-granular leaderboard 구축

<br>

# 1. Introduction

**Paradigm Shift**

- TSFM 등장으로 TSF evaluation 방식 변화
  - dataset-centric → **task-centric**
  - 개별 dataset 학습·평가 → **zero-shot** generalization 중심 

<br>

**기존 Benchmark의 구조적 한계 (4가지)**

- **(1) Legacy-Constrained Data Coverage**
  - 기존 benchmark가 **소수 legacy dataset에 의존**
  - **성능 plateau 현상**
  - **Data leakage 위험** 증가

- **(2) Compromised Data Integrity**
  - Quality control 미흡
  - Extreme outlier, excessive missing value 포함 variate 존재
- **(3) Misaligned Task Formulation**
  - 현실과 동떨어진 기계적 forecasting setup
  - Predictable horizon, variate predictability 미고려
- **(4) Limited Analysis Perspective**
  - Meta label 기반 **dataset-level** evaluation
  - Domain/frequency 단위 평균 성능 제시
  - Temporal structure 관점 분석 부족
  - Error metric(MAE, MASE 등)이 structural fidelity 반영 못함

<br>

**TIME Benchmark 제안**

- ***TSFM에 적합한 task-centric benchmark*** 설계
- 두 축에서 발전:
  - (1) **Benchmark** construction
  - (2) **Evaluation** perspective

<br>

**Benchmark Construction**

- **50개 fresh dataset** 수집 
  - 기존 benchmark에서 거의 사용되지 않음
- **Human-in-the-loop** data curation pipeline
  - 자동 screening + human refinement
- LLM + domain knowledge 기반 task validation
- Frequency, horizon을 real-world operational requirement에 맞춰 설계
- **Strict zero-shot** evaluation 보장

<br>

**Evaluation Perspective**

- Dataset-level → **Pattern-level** evaluation
- **STL decomposition** 기반 structural feature 추출
- **Binary encoding**으로 temporal pattern 표현
- **Pattern-specific** aggregation 통해 cross-dataset generalization 분석

<br>

**Contributions**

- **50 dataset, 98 task**로 구성된 **task-centric** benchmark
- **Leakage-free zero-shot** evaluation 환경 제공
- **Pattern-level** benchmarking으로 diagnostic & generalizable insight 제공

<br>

# **2. Related Work**

## (1) Time Series Forecasting Benchmarks

a) **초기 TSF 연구**

- 표준 benchmark 부재
- 각 논문이 **4–6개 dataset 개별** 선택
- unified evaluation protocol 없음

<br>

b) **초기 대규모 Benchmark**

- **M4 competition** 등장
- 대규모 univariate benchmark 제공
- DL 모델 비교 가능 환경 마련

<br>

c) **Transformer 기반 Benchmark**

- LSF 등장 (**long-horizon forecasting** 표준화)
- TSF community에서 가장 널리 사용
- **Monash** archive → 다양한 domain dataset 제공
- 이후 benchmark 다수가 LSF + Monash에 의존

<br>

d) **Domain-Specific Benchmark**

- M5 (sales)
- CloudOPS (cloud metrics)
- LibCity (urban transportation)

<br>

e) **TSFM 등장 이후**

- **Zero-shot evaluation** 목적의 대규모 benchmark 증가
- **Diverse** domain + **Large-scale** coverage 강조
- **Universal** forecasting capability 평가 목적

<br>

**최근 Benchmark의 한계**

- **(1) Data Novelty 부족**
  - 기존 dataset 재조합 중심
  - Genuinely new data source 부족
- **(2) Data Integrity & Task Formulation 문제**
  - 검증되지 않은 data flaw 존재
  - Real-world와 분리된 mechanical setup
- **(3) Conventional Evaluation**
  - Dataset-level metric 집계 중심
  - Metadata 기반 grouping (domain 등)
  - Intrinsic temporal pattern 미반영
  - Universal dynamics에 대한 generalizable insight 부족

<br>

## (2) Time Series Features

**배경**

- Time series는 NLP/vision과 달리 **직관적 semantic 부재**
- **Heterogeneity가 높아** 해석 및 categorization 어려움

<br>

**초기 TS feature 활용**

- 주로 classification task에 활용
- Synthetic data generation에 사용

<br>

**Forecasting 맥락에서의 활용**

- 일부 feature 선택 후 PCA로 시각화
- Dataset 내부 variate 분포 분석 목적

<br>

**최근 Benchmark에서의 활용**

- TS feature 기반 high-level property 정의
  - **trend**
  - **seasonality** 
- Dataset coverage 분석에 사용

<br>

**한계**

- 여전히 **dataset-level 분석**에 머묾
- Cross-dataset variate에 대한
  - **pattern-specific** performance 분석 부족
  - model capability에 대한 **generalizable insight 제한적**

<br>

# 3. Preliminary

## (1) Time Series Forecasting Benchmarks

**TSF Benchmark 정의**

- Forecasting method를 체계적으로 evaluation하는 unified platform
- 계층적 구조로 formalization

<br>

**Hierarchy 구성**

- **(1) Benchmark**

  - **여러 task**로 구성
  - 공통 evaluation protocol 공유
  - 다양한 analysis perspective 제공

- **(2) Task**

  - **(특정 dataset, 특정 prediction horizon)** 조합

- **(3) Dataset**

  - **하나 이상의 series** 포함
  - 동일 데이터라도 **sampling frequency가 다르면 별도 dataset**으로 간주

- **(4) Series**

  - Univariate 또는 multivariate 가능
  - 같은 dataset 내 series는 variate set 공유
  - Temporal length는 서로 다를 수 있음

- **(5) Variate**

  - **단일** time-dependent variable

  - **UTS의 전체 시계열** or **MTS의 channel**

  - 본 benchmark에서는 **모든 variate를 prediction target**으로 사용

    (단, Exogenous covariate는 제외)

- **(6) Testing Window**

  - Forecast 평가를 위한 연속 구간
  - 길이는 일반적으로 **prediction horizon과 동일**

<br>

## (2) Forecasting Task

**Task 정의**

- $T = (\mathcal{D}, H)$,
  - $\mathcal{D}$: Dataset
  - $H$: Prediction horizon

<br>

**Dataset 구성**

- $\mathcal{D} = \{X^{(i)}\}_{i=1}^N$,
  - $N$개의 time series 포함
  - 각 series $X \in \mathbb{R}^{L \times D}$
    - $L$: temporal length
    - $D$: variate 수

<br>

**Series 유형**

- $X = [x_1, x_2, ..., x_D]$
  - $D = 1$ → univariate (UTS)
  - $D > 1$ → multivariate (MTS)

<br>

**Test Set 구성**

- 각 series의 마지막 $L_{test}$ 구간을 test set으로 분리

<br>

**Sample 생성 방식**

- Non-overlapping rolling window 사용
- Stride = $H$
- Series당 sample 수: $W = \lfloor L_{test} / H \rfloor$.

<br>

**k번째 Sample 정의**

- $1 \le k \le W$
- 시작 index: $t_k = (k-1)H$
- Testing window: $X_{t_k : t_k + H} \in \mathbb{R}^{H \times D}$

<br>

## (3) Time Series Pattern

**정의**

- TS pattern = **단일 variate**의 intrinsic temporal characteristic
- raw data를 pattern으로 abstraction
- cross-dataset evaluation을 위한 연결 고리 역할
- pattern-level analysis 가능하게 함

<br>

**Pattern Representation**

- variate x에 대해
- F = (F_1, ..., F_K) 로 정의
- K개의 time series feature로 구성된 벡터

<br>

**의미**

- 통계적·구조적 특성을 정량화
- series의 structural nature 표현

<br>

**Feature Selection**

- 구체적 feature 구성은 Section 5에서 정의

<br>

# **4. Benchmark Construction**

## (1) Data Curation

**Motivation**

- **Legacy** dataset: **pre-training corpus에 포함**되었을 가능성 존재
- Inadvertent / malicious contamination 위험
- Benchmark evaluation 신뢰성 저하 문제

→ data novelty 우선 확보

<br>

### a) **Data Sources (4가지)**

- 정부 공식 portal의 public statistics
- 산업·학계 파트너와의 협업을 통한 real-world data
- open-access website 데이터
- forecasting competition 데이터

<br>

### b) **Manual Curation Pipeline**

- **(1) Eligibility Check**
  - 연속적 time series 여부 확인
  - 유효한 timestamp 존재 여부
  - Regular sampling frequency 검증
  - Forecastable application context 존재 여부 확인
- **(2) Metadata-Based Filtering**
  - Frequency 및 time span 점검 $\rightarrow$ Frequency 대비 너무 짧은 series 제거
  - Application context와 무관한 variate 제거
- **(3) Visualization-Based Inspection**
  - 명확한 temporal pattern 없는 variate 제거
    - Constant sequence 등
  - Excessive missing value 포함 variate 제거
  - 전체 구간 품질이 낮을 경우
    - 신뢰 가능한 sub–time span 추출 가능성 검토
- **(4) Dataset Organization**
  - Section 3의 hierarchy 구조에 맞게 구성
  - 공통 time span 정렬 후 multivariate series로 결합 가능
  - 그렇지 않으면 multiple univariate series로 유지

<br>

## (2) Automatic Screening

**목적**

- Raw dataset의 ***fine-grained quality*** profiling
- Series/variate 즉시 삭제하지 않고 **Quality Summary** 생성
- 5단계 sequential pipeline

<br>

### **자동화 단계**

1. **Timestamp Rectification**
   - Missing / misaligned timestamp 확인
   - Missing timestamp 채움
   - Misaligned timestamp → nearest standard timestamp로 교정 (frequency 기반)
2. **Rule-based Validation**
   - 사전 정의 rule에 따라 자동 검증
   - **Missing rate, length threshold** 위반 variate flag
   - Value dominance로 **temporal dynamics 없는** uninformative series 식별
3. **Statistical Test**
   - Ljung-Box test 적용 → **white noise variate 제거** 후보 flag
4. **Extreme Outliers Removal**
   - Local IQR filter로 극단값 제거
   - time point $t$, window $w$: 오류 if outside $[m_w - k·IQR_w, m_w + k·IQR_w]$
   - 오류 값 → 이전 valid observation으로 대체
   - $k=9$: genuine spike 유지, technical error만 제거
5. **Correlation Check**
   - Dataset 내 variate pairwise correlation 계산
   - Correlation > threshold → high collinearity flag → expert review

<br>

**결과**

- 모든 진단 결과를 dataset-level **Quality Summary**로 집계
- 최종 human decision-making에 제출

<br>

## (3) Human Decision Making

### a) Dataset Finalization via Review

**목적**

- **Quality Summary** 기반으로 ***human judgment 적용***
- **자동화로 해결 불가**한 quality ambiguity 해결
- Dataset structure 최종 확정

<br>

**Review Process**

- (1) Domain knowledge + (2) LLM-generated insight 활용
- Flagged variate 평가
  - genuine data corruption vs. meaningful domain characteristic 구분

<br>

**예시**

- Car park availability: high correlation → redundancy → 제거 가능
- Macroeconomic indicator: high correlation → 구조적 dependency → 유지

<br>

**Granularity 결정**

- Series-level 제거 (문제 있는 series)
- Variate-level 제거 (예측에 부적합 variate)

<br>

**결과**

- Dataset $\mathcal{D}$: High data integrity 유지
- Real-world application과 alignment 보장

<br>

### b) Context-Aligned Task Formulation

**목적**

- Forecasting task는 **real-world operational requirement 반영** 필요
- 단순 default 설정 사용 지양

<br>

**설정 방법**

- Data frequency + task configuration → **Domain expertise + LLM 분석 기반** 결정

<br>

**Prediction Horizon (H)**



- High-frequency dataset: Short / Medium / Long horizon 세 가지 정의
- Low-frequency or constrained dataset: 하나의 operationally viable horizon



**Test Length (Ltest)**



- practical cycles와 alignment (예: 전체 seasonal period 포함)
- 평가 framework가 실제 적용 가능성 중심으로 설계







