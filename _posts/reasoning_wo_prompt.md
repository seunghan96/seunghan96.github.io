# Chain-of-Thought Reasoning without Prompting

Wang & Zhou, Google DeepMind, arXiv 2402.10200v2, 2024



***“Prompt 없이도 LLM이 본래 갖고 있는 추론 능력을 끌어낼 수 있다”***







# 1. Background

- 지금까지 LLM의 **추론(Reasoning)** 성능은 대부분 **CoT prompting**을 통해 드러남

  - e.g., *“단계를 나눠서 풀어라”* 같은 지시문을 Prompt에 넣기

- 하지만 이는 **Prompt engineering**에 의존

  $\rightarrow$ LLM의 “본질적 추론 능력”을 정확히 평가하기 어려움

<br>

# 2. Key Idea

**질문 그대로**(QA 형식) + Decoding만 바꿔봄.

- 보통은 **Greedy decoding (top-1 계속 선택)**을 하는데,
- 여기서 **"top-k 대안 경로"**를 살펴보면, **자연스럽게 CoT 추론 경로가 숨어 있음**!!

<br>

즉, **추론 경로**를 내부에 이미 가지고 있지만, greedy decoding 때문에 잘 드러나지 않았던 것일 뿐!!!

$\rightarrow$ CoT-decoding

<br>

# 3. CoT-Decoding

1. **Top-k branching**

   - 첫 번째 Decoding 단계에서 **"top-k 후보"**를 여러 개 뽑고 각각 경로를 탐색
   - Greedy path는 종종 오답을 내지만, **다른 경로에는 CoT 추론이 존재**함을 확인

2. **Confidence 기반 선택**

   - **CoT가 "있는" 경로**일수록 **"최종 답의 token 확률 차이(Δ)"**가 커서 **더 확신(confidence)**을 보임
   - Idea) 이를 활용해 CoT-path를 선택하자!!

   





------





## **📊 주요 실험 결과**





- **수학 추론(GSM8K), 상식 추론(year parity), MultiArith 등**에서 평가.

- Greedy decoding에 비해 **큰 성능 향상**:

  

  - PaLM-2 Large 기준, GSM8K 정확도 34.8% → 63.2% 
  - year parity 과제는 거의 **95% 정확도**에 도달 .

  

- Mistral, Gemma 등 오픈 모델에도 적용 가능 → reasoning 성능 2~3배 향상 .

- Instruction-tuned 모델에도 추가 이득이 있음 (예: Mistral-7B instruct, +7~30% 정확도) .





------





## **📌 기여점**





1. **Prompt 없이도 LLM은 CoT 추론 경로를 본래 가지고 있음**을 처음으로 보임.
2. **Decoding 전략만 바꿔서** LLM의 내재적 추론 능력을 평가할 수 있게 함.
3. **Confidence 기반 CoT-decoding** 제안 → Prompt 없이도 Self-consistency 같은 효과 달성.
4. 모델 크기, 튜닝 여부와 관계없이 일관된 향상.





------





## **⚠️ 한계와 논의**





- top-k 여러 경로를 탐색해야 하므로 **추가 연산 비용** 발생.
- open-ended 답변이나 정답 span 식별이 어려운 경우엔 confidence 측정이 덜 정확할 수 있음.
- 현재는 **첫 token 분기**에 초점을 맞췄는데, 다른 단계에서도 branching 탐색이 가능할지 연구 필요 .





------





## **✅ 요약 한 줄**





> **“LLM은 Prompt가 없어도 CoT 추론 경로를 본래 품고 있으며, 단지 greedy decoding에 가려져 있었다. Decoding을 바꾸면 이 추론이 드러난다.”**



------



혹시 원하시는 요약 스타일이



- (a) Motivation/Method/Experiment/Conclusion 형식의 학술 요약,

- (b) 5 bullet 핵심 정리(짧게),

  둘 중 어떤 쪽일까요?