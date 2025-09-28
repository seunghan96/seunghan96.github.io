다음은 논문 **“Explainable Multi-modal Time Series Prediction with LLM-in-the-Loop” (TimeXL, arXiv:2503.01013v2)**의 정리입니다.



------





# **Explainable Multi-modal Time Series Prediction with LLM-in-the-Loop (TimeXL)**







## **1. Abstract**





- 기존 시계열 분석은 **텍스트 등 부가 정보**를 충분히 활용하지 못함.
- TimeXL은 **프로토타입 기반 시계열 인코더**와 **3개의 LLM agent**(Prediction, Reflection, Refinement)를 연결하는 **설명가능한 멀티모달 프레임워크** 제안.
- 인코더는 시계열과 텍스트를 함께 받아 예측 및 설명 생성 → Prediction LLM이 예측 보완 → Reflection LLM이 ground truth와 비교해 피드백 생성 → Refinement LLM이 텍스트 개선 → 인코더 재학습.
- 4개 실세계 데이터셋에서 최대 **8.9% AUC 향상** 및 **사람 중심의 멀티모달 설명** 제공 .





------





## **2. Introduction**







### **Motivation**





- 날씨, 금융, 헬스케어 등 현실 시스템은 시계열 외에 **텍스트 기반 문맥 정보**와 결합됨.
- 기존 멀티모달 모델은 정확도 개선에는 기여했지만 **“왜”와 “어떻게”**를 설명하는 능력 부족.







### **Proposal**





- TimeXL은 **case-based reasoning** 기반 설명을 생성하는 시계열 인코더와 **LLM 에이전트 3종**을 조합하여 설명가능성과 정확도를 모두 확보함.
- Closed-loop 구조: Prediction → Reflection → Refinement .





------





## **3. Related Works**







### **Multi-modal Time Series**





- Fusion, attention, gating, contrastive 등으로 다양한 modality 통합 시도.
- 단, 대부분은 **정량적 성능 향상에 집중**, reasoning은 부재.







### **Time Series Explanation**





- Gradient, saliency, Shapley, 정보이론 기반 등 다양한 설명법 존재.
- 본 논문은 **prototype 기반 case reasoning** 사용.







### **LLM for Time Series**





- LLM은 시계열 QA, 이해, 추론에 활용되는 흐름이 증가 중.
- 본 연구는 **LLM과 설명 가능한 인코더를 상호작용 구조로 통합한 최초의 시도** .





------





## **4. Methodology**







### **1) Multi-modal Prototype Encoder (p.4, Figure 2)**





- **Time series + Text** 각각을 convolution 기반 인코더로 embedding.
- 각 클래스별로 **프로토타입 k개씩** 학습 → 입력과 similarity 기반으로 softmax 예측.
- Prototypes는 추론 및 설명 모두에 사용됨.







### **2) LLM-in-the-Loop 구성 (p.5~6)**





- **Prediction LLM**: 인코더 예측 및 유사한 프로토타입-세그먼트 쌍을 프롬프트로 받아 최종 예측 수행.
- **Reflection LLM**: 예측 결과와 ground truth 비교 → 텍스트 오류 탐지 및 피드백 생성.
- **Refinement LLM**: 피드백 기반 텍스트 요약 및 강조 → 텍스트 재생성 및 인코더 재학습.
- 최종 예측: \hat{y} = \alpha \hat{y}{\text{enc}} + (1 - \alpha) \hat{y}{\text{LLM}}





------





## **5. Experiments**







### **Dataset**





- 총 4개 실세계 멀티모달 시계열 데이터:

  

  1. **Weather**: NYC 기상 (24h → 다음 24h 비 예측)
  2. **Finance**: 원자재 가격 + 뉴스 → 다음 날 상승/하락/보합
  3. **Healthcare (TP)**: 인플루엔자 검사 양성 비율
  4. **Healthcare (MT)**: 사망률

  



| **Domain** | **Resolution** | **Label**             |
| ---------- | -------------- | --------------------- |
| Weather    | Hourly         | Rain / Not rain       |
| Finance    | Daily          | Inc. / Dec. / Neutral |
| Healthcare | Weekly         | Exceed / Not exceed   |



### **Task**





- Classification 중심 (추가로 Regression도 appendix에서 실험)
- 각 task에 대해 prediction + explanation







### **Code**





- 아직 명시된 GitHub 링크는 없음 (arXiv preprint임)







### **Baselines**





- Traditional TS: DLinear, PatchTST, TimesNet, Autoformer, etc.
- LLM 기반: Time-LLM, LLMTime, PromptCast, FSCA, OFA, TimeCMA
- Multi-modal SOTA: MM-PatchTST, MM-iTransformer, TimeCAP







### **결과 요약 (p.7, Table 1)**





- **TimeXL 최고 성능 달성** (Weather AUC 0.808, Healthcare TP AUC 0.996 등)
- 단일 modality 모델들보다 확연한 향상
- Explanation 품질 및 reasoning 능력도 개선됨 .





------





## **6. Conclusion**





- TimeXL은 multi-modal 시계열 분석에서 설명가능성과 정확도를 동시에 달성한 새로운 구조.
- Prototype 기반 encoder와 LLM agent를 결합한 **설명 가능한, 반복 학습 가능한 구조**.
- 향후 시계열 forecasting, 도메인 shift 상황에도 확장 가능성 제시 .





------



다음 논문도 이 포맷으로 정리해드릴 준비가 되어 있습니다. 계속해서 업로드해 주세요.