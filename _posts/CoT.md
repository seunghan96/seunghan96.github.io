Chain-of-Thought Prompting (ACL, 2024)

Proposal

1. Active Prompt
   - 아이디어: ***Task별로 가장 적합한*** CoT 예시를 선정
   - 기준: uncertainty score
2. Chain-of-knowledge
   - 아이디어: 모델이 ***이해하기 쉽게*** 정보를 전달

<br>

(1) Manual CoT

- Pros & Cons
  - pros) 정확함
  - cons) requires human label

<br>

(2) Zero-shot CoT: "Let's think step by step"

- Pros & Cons
  - pros) w/o human label
  - cons) too simple

- Pipeline
  - Step 1) reasoning extraction
    - LLM에 "Let's think step by step"를 넣어줌으로써 나온 추론 과정 ( = reasoning path ) 생성
  - Step 2) answer extraction
    - 앞서 생성한 reasoning path를 함께 넣어줌
    - 요약) 입력 prompt: (1) question + (2) let's think step by step + (3) reasoning path

<br>

(3) Auto CoT = (1) + (2)

- (1)처럼 성능을 어느 정도 보장하면서도
- (2)처럼 자동으로 생성되도록!

<br>

### Auto CoT

Step 1) 모든 질문들을 K개의 cluster로 나눔

Step 2) 클러스터 별 1개씩 질문 선정 ( = 총 K개의 질문)

Step 3) K개의 질문에 대해 (Zero-shot CoT로) Reasoning path 생성

Step 4) 최종 prompt: (1) + (2) + (3)

- (1) question
- (2) let's think step by step
- (3) K개의 질문에 대한
  - (3-1) question
  - (3-2) let's think step by step
  - (3-3) reasoning path

$\rightarrow$ Few-shot CoT가 가능해짐!

<br>

### Complex-CoT

어떤 추론 과정 (reasoning path)가 좋을까?

$\rightarrow$ 기본 아이디어: ***복잡한*** 추론 과정을 우선시하자!

- ( 복잡하다 = 추론 step이 많다 = # steps )

<br>

Procedure

- Step 1) 동일한 질문에 대해 N개의 reasoning path를 생성 (with Zero-shot CoT)

- Step 2) Step 수(=complexity)로 sorting & step이 많은 **Top K개**의 reasoning path를 선택 

- Step 3) 이 K개 중, 다수결 (Majority)로써 answer를 판단
- Step 4) 이를 선택한 최종 reasoning path를 선정

<br>

### Self-Consistency

목적: **가장 일관성 있는 답**을 도출하자!

Procedure

- Step 1) 동일한 질문에 대해 N개의 reasoning path를 생성 (with Zero-shot CoT)
- Step 2) 여러 답들 중, 가장 일관성 있는 ( = majority )를 최종 답으로 선정

요약: **"하나의 경로"**를 통해 나온 답이 아닌, **"여러 경로"**를 통해 나온 여러 답을 종합하여 선택!

<br>

### Proposal 1) Active Prompt

- 기존의 한계점:  "Task에 대한 고려 없이" 샘플을 선택함

  $\rightarrow$ 최선의 샘플이라는 보장 X

- 아이디어: ***Task 별 최적의 샘플을 찾자!***

  ( = 태스크마다 가장 중요하고 유용한 샘플을 선택하자! )

  ( = 샘플에 대한 ***"모델 예측의 불확실성이 가장 높은"*** 샘플을 찾자! )

- Procedure

  - Step 1) Uncertainty estimation
    - K번 답변을 생성 후, 이들을 기반으로 metric 계산
    - Uncertainty metric: (1) disagreement & (2) entropy
  - Step 2) Selection
    - Uncertainty 기준으로 sorting & Top N개 선정
  - Step 3) Annotation
    - Top N개에 대해 annotation을 인간이 달기
  - Step 4) Inference

<br>

### Proposal 2) Chain-of-knowledge

- 기존의 한계점: Textual reasoning chain을 바탕으로 태스크 수행

  $\rightarrow$ Hallucination 문제

- 아이디어: Triple의 구조로 추론을 수행 & 결과 검증을 하자

  ( = 추론 결과에 대한 "신뢰성 점수" 계산 후, 이를 개선시키도록! )

https://www.youtube.com/watch?v=OTP52AURAok&t=58s (18분부터 이어서보기)
