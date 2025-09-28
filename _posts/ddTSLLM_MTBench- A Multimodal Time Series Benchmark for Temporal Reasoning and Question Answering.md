# MTBench: A Multimodal Time Series Benchmark for Temporal Reasoning and Question Answering

- https://arxiv.org/pdf/2503.16858

<br>

# **1. Abstract**

**"Text와 TS 간의 관계"**를 평가하는 기존 Benchmark:

$\rightarrow$ **Cross-modal reasoning**을 충분히 다루지 못함!

<br>

### **MTBench**

[Dataset] 

- 금융 및 기상 도메인
- **(TS + 뉴스 Text) 쌍**으로 구성된 최초의 **Multi-task Multimodal Benchmark**

<br>

[Task] 

- a) Forecasting
- b) Semantic Trend Analysis
- c) Technical Indicator Prediction
- d) News-driven QA

<br>

LLM (GPT-4o, Claude, Gemini 등)은 여전히 **causal 추론, 장기 의존성**, **모달 간 통합**에 약점을 보임 .

<br>

![image-20250827152205587](/Users/seunghan96/Library/Application Support/typora-user-images/image-20250827152205587.png)

<br>

# **2. Introduction**

## (1) Motivation & Proposal

- **정량적 추세(TS)**와 **정성적 설명(Text)**이 본질적으로 얽혀 있음.
- 기존 (TS & Text) Benchmark:

  - Forecasting 중심 & ***Reasoning 부족!***
- Proposal: MTBench
  - **TS과 의미적으로 연결된 Text**를 함께 제공 $\rightarrow$ Reasoning task 평가 가능.
  

<br>

# **3. Related Works**

(1) **기존 multimodal benchmark** (Time-MMD, ForecastBench, TimeseriesExam 등) 

- 해상도/도메인/태스크 다양성 부족.

(2) **기존 금융 benchmark**

- 대다수 단일 modality만 포함하거나 트위터 기반.

(3) **기존 날씨 benchmark**

- Numerical 중심이며 text reasoning은 부족.

<br>

$\rightarrow$ MTBench는 **도메인 전문적 뉴스**와 **실제 TS**을 시간적으로 정렬된 구조로 구성하여 이전 한계를 보완함 .

<br>

# **4. Methodology**

## (1) Dataset 구성

### a) **Finance**

- **20만 개 금융 뉴스** URL 수집 

  → GPT-4o로 category, sentiment, temporal label 등 tagging

- TS은 뉴스에 언급된 **종목의 주가 (5분 or 1시간 단위)**로 수집.

- Pair 총 **20,000쌍** 구성 (short/long-term forecast 각각)

- *Consistent vs Misaligned 뉴스*로 나누어 모델의 판별 능력도 평가 가능

<br>

### b) **Weather**

- 50개 공항 날씨 데이터 (GHCN-H, 2003~2020, 1시간 단위)
- 스톰 이벤트와 인근 공항 데이터 정렬 + LLM 기반 synthetic 뉴스 생성
- 총 **2,000쌍** (각 스테이션당 40개)

<br>

![image-20250827152229192](/Users/seunghan96/Library/Application Support/typora-user-images/image-20250827152229192.png)

<br>

## (2) Task 구성

(a) **Forecasting (Regression)**

- Finance: 7일 or 30일 입력 → 1일 or 7일 예측
- Weather: 7일 or 14일 입력 → 1일 or 3일 예측

<br>

(b) **Trend Analysis (Classification)**

- 가격 변화율을 binning하여 3/5-class 예측

<br>

(c) **Technical Indicator Prediction**

- Finance: MACD, Bollinger Band 예측
- Weather: 최고/최저/차이 예측

<br>

(d) **News-driven QA**

- Correlation Prediction (3/5-class)
- Multi-choice QA (News + TS 기반 reasoning) 

<br>

![image-20250827152253343](/Users/seunghan96/Library/Application Support/typora-user-images/image-20250827152253343.png)

<br>

# **5. Experiments**

## (1) **Baseline Models**

- GPT-4o, Claude 3.5, Gemini 2.0, LLaMA3.1-8B, DeepSeek-Chat, OpenAI-o1

<br>

## (2) **주요 결과 요약**

### a) Forecasting

**Text 추가 시 ...**

- **평균 9.78% (finance), 6.63% (weather) 성능 향상**
  - 장기 예측은 모든 모델에서 성능 저하됨

- LLM은 종종 출력 길이 제약을 정확히 따르지 못함

<br>

### b) Trend Prediction

- 과거 추세 분석 > 미래 추세 예측 (정확도 차이 큼)
- Text 추가 시 28개 중 25개 case에서 정확도 향상
- 회고 분석에서는 간혹 성능 하락 (text 활용 실패)

<br>

### c) Indicator Prediction

- Text는 특히 Bollinger Band에서 도움이 됨
- OpenAI-o1 모델이 대부분 가장 낮은 MSE 기록

<br>

### d) News-driven QA

- 30일 long-term 설정이 7일보다 오히려 더 쉬움
- MCQA에서는 Claude, DeepSeek이 가장 높은 정확도 달성
- 모델들은 대부분 **긍정적인 correlation bias**를 보임 → 약하거나 음의 상관관계는 과소 인식 

<br>

# **6. Conclusion**

- MTBench는 **Text-TS 통합 reasoning 능력**을 평가할 수 있는 최초의 대규모 Benchmark.

- 기존 모델은 surface-level task에는 강하지만 **장기 추론, 인과 해석, 다중 모달 통합**에 여전히 한계 존재

  

