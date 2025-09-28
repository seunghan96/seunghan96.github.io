# Empowering Time Series Analysis with Synthetic Data: A Survey and Outlook in the Era of Foundation Models







## **1. Abstract**





- 시계열 Foundation Models (**TSFMs**)과 LLM 기반 모델(**TSLLMs**)은 task-agnostic generalization과 contextual 이해를 가능케 함.
- 그러나 이러한 모델의 성공은 **대규모, 다양하고 고품질의 데이터셋**에 의존함.
- **Synthetic data**는 이 한계를 극복할 수 있는 대안으로 주목됨.
- 본 논문은 TSFM 및 TSLLM의 학습 전주기(pretraining, finetuning, evaluation)에 걸친 synthetic data의 활용을 체계적으로 정리하고 미래 방향을 제시함 .





------





## **2. Introduction**







### **Motivation & Proposal**





- 기존 시계열 모델은 특정 task나 도메인에 특화된 구조였지만, 최근에는 **zero-shot 가능한 TSFM**과 **텍스트 기반 추론이 가능한 TSLLM**으로 진화.

- 문제는:

  

  1. 규제 및 저작권 이슈로 인해 대규모 실데이터 확보 어려움
  2. 도메인 편향과 품질 저하
  3. **시계열-텍스트 쌍 데이터의 부족**

  

- **Synthetic data**는 스케일러블하고, bias-free하며, 다양성과 통제력을 가짐.

- 이 서베이는 synthetic data의 생성/활용법을 **TSFM과 TSLLM으로 나누어** 전주기에 걸쳐 정리함 .





------





## **3. Related Works**





- 기존 서베이들과 비교 시 본 논문만이 **TSFM, TSLLM, synthetic data 생성 및 활용 모두를 포괄**함 (p.2 Table 1) .





------





## **4. Methodology**







### **Time Series Foundation Models (TSFMs)**





- **사용 목적**: forecasting, classification 등 classical task 중심

- **Synthetic data generation** 방식:

  

  - **ForecastPFN**: trend × seasonality × noise (multiplicative)
  - **TimesFM**: piecewise linear + ARMA + sine/cos
  - **Chronos**: Gaussian Process kernel 조합 (KernelSynth)
  - **Moment**: sinusoid 기반 simple pattern 학습용 

  

- **Pretraining 활용 예시** (p.4 Table 3):

  

  - ForecastPFN, TimePFN, Mamba4Cast, Chronos, TimesFM 등
  - 일부는 **pure synthetic**, 일부는 **real+synthetic 혼합**
  - **Chronos**: synthetic 비중 10%일 때 성능 향상, 그 이상은 감소

  

- **Evaluation 활용**:

  

  - Moment, Wiliński et al.: TSFM의 hidden representation 평가
  - Potosnak et al.: reasoning 평가용 synthetic data
  - Freq-Synth: frequency generalization 진단용 sin wave 활용 

  

- **한계점**:

  

  - Pretraining에서 ad-hoc 방식, systematic gap 채움 부족
  - 대부분 statistical 방식, diffusion 등 data-driven 생성법 부족
  - Fine-tuning에 synthetic data 활용 거의 없음 

  





------





### **Time Series LLMs (TSLLMs)**





- **사용 목적**:

  

  1. Forecasting (context 활용)
  2. Reasoning: QA, MCQA, Captioning, Explanation 등

  

- **주요 분류 체계** (p.5 Figure 1):

  

  - **텍스트 생성 방식**: Template / LLM / Web-crawled
  - **데이터 구성**: Real-TS + Syn-Text / Syn-TS + Syn-Text 등

  

- **Pretraining 활용**:

  

  - **ChatTS**: TS encoder 추가, synthetic TS + 텍스트 쌍으로 학습
  - **TempoGPT**: TS를 discrete token으로 quantize하여 LLM과 alignment
  - **Chow et al.**: Mistral 기반 encoder + QA pair 학습 

  

- **Finetuning 활용**:

  

  - 대부분 instruction-following 구조로 QA, Reasoning 등 학습
  - **ChatTime**, **Insight Miner** 등은 LLaMA 또는 GPT 기반 모델 finetune

  

- **Evaluation 활용**:

  

  - synthetic 데이터셋 기반 MCQA, reasoning 등 벤치마크 생성
  - **TimeSeriesExam**, **Merrill et al.**, **XForecast**, **LLMTime** 등

  

- **한계점**:

  

  - Synthetic TS의 **현실감 부족**
  - TS-Text alignment의 정확성 부족
  - Evaluation 시 quality 보장 어려움 

  





------





## **5. Experiments**





- (해당 논문은 서베이이므로 별도의 실험은 없음)
- 다만, 주요 모델들의 synthetic data 활용 여부 및 양, 목적은 p.4~6의 Table 3, 4에 정리됨.





------





## **6. Conclusion**





- Synthetic data는 TSFM과 TSLLM 발전에 필수적이며, 특히 pretraining 및 benchmarking에서 핵심 역할 수행.

- 그러나 여전히 다음과 같은 연구 과제 존재:

  

  - 패턴 채움을 위한 **전주기적 synthetic data lifecycle**
  - **data-driven generation** 방식 확장 (e.g., diffusion)
  - **self-improving synthetic generation loop** (FM이 synthetic data 만들어 스스로 학습) 

  





------



📌 *이 논문은 실험 논문이 아닌 고도화된 서베이로, 지금까지 등장한 대부분의 TSFM/TSLLM을 synthetic 관점에서 정리해 줍니다. 다음 논문을 주시면 계속해서 정리해드릴게요.*