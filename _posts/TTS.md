# Test-time scaling (TTS)

## (1) 개요

(핵심 아이디어) 높은 성능을 위해, **"더 오래 + 더 넓게" 생각**하자!

- a) 더 **"많은"** 샘플 (토큰) 생성
- b) 더 **"깊은"** 탐색

<br>

TTS 설명

- (1) **정의**:  " **inference(test) 시점에 추가 연산을 더 쓰거나 배치를 바꿔** " 성능을 끌어올리는 방법론
- (2) 기존 vs. TTS
  - (기존) Train의 compute를 강조
  - (TTS) "Inference"의 compute를 강조

<br>

## (2) 사용 이유

**난이도가 높은 inference**에서 한 번의 forward: ***불안정***

$\rightarrow$ **"여러 경로"를 탐색/표결/재평가**하면 더 좋아질 수 있음을 확인!

<br>

e.g., OpenAI **o1** 계열

- **inference 시 생각(연산) 시간을 더 들여** 정확도를 올리는 접근

<br>

## (3) TTS 방법

### a) **Breadth (가로 확장)** – 다중 샘플 & 표결

- **Self-Consistency** (간단하지만 강력):
  - Procedure
    - Step 1) 같은 문제를 temperature sampling으로 **N번** 풀기
    - Step 2) **다수결/스코어**로 최종 답 채택
  - Reference: 
    - *Self-Consistency Improves Chain of Thought Reasoning in Language Models* (https://arxiv.org/pdf/2203.11171)

![figure2](/assets/img/llm/img803.png)

<br>

- **Best-of-N + Reranking**:
  - **Self-Consistency** 자체가 일종의 Best-of-N 아이디어를 포함
  - 차이점? 더 일반적으로 **후보군 생성 → "검증/채점기로 reranking"**
  -  Procedure
    - Step 1) 후보 N개 생성 
    - Step 2) 외부 채점기 (Rule/Scorer/LLM-judge)로 최고 후보 선택.

  - Reference
    - *STaR: Bootstrapping Reasoning With Reasoning* (https://arxiv.org/pdf/2203.14465)
      - 여러 reasoning 후보를 생성 + 정답/검증 기준에 맞는 경로를 **재선택하여 학습 데이터에 반영**

    - *Self-Refine: Iterative Refinement with Self-Feedback* (https://arxiv.org/pdf/2303.17651)
      - 후보 답안을 생성한 뒤 모델 스스로 피드백을 주고, 그에 따라 **수정/재랭킹**하는 구조
    - *Competition-Level Code Generation with AlphaCode* (https://arxiv.org/pdf/2203.07814)
      - 수십만 code 후보 생성 후, **test case 채점기**로 성능 좋은 후보만 채택


![figure2](/assets/img/llm/img804.png)

![figure2](/assets/img/llm/img805.png)

![figure2](/assets/img/llm/img806.png)

<br>

### b) **Depth (세로 확장)** – 더 깊게 생각하기

- **더 긴 inference 토큰** (예: `reasoning.effort`을 높이거나, 단계별 사고를 장려)

```python
resp = client.responses.create(
    model="o1",
    input=[{"role": "user", "content": "어떤 수학 문제"}],
    reasoning={"effort": "high"},   # ← 깊게 생각
    max_output_tokens=800,
)
```

<br>

- **Longer Inference / Chain-of-Thought Scaling**

  - Procedure
    - Step 1) 모델이 답을 내기 전 **더 많은 reasoning tokens**을 생성하도록 유도
    - Step 2) 긴 사고 과정을 통해 정답률을 향상
  - Reference
    - *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* (https://arxiv.org/pdf/2201.11903)
      - LLM이 **단계별 추론(CoT)**을 거치면 복잡한 수학/논리 문제에서 성능이 급상승
    - *Scaling Instruction-Finetuned Language Models* (a.k.a. FLAN-PaLM, https://arxiv.org/pdf/2210.11416)
      - 모델 크기뿐 아니라 **추론 토큰 길이**를 늘릴 때 성능이 함께 스케일링됨을 보고

  

![figure2](/assets/img/llm/img807.png)

![figure2](/assets/img/llm/img808.png)

<br>

- **Deliberate Decoding / Deliberation Networks**

  - Procedure

    - Step 1) 모델이 초안(draft) 답변을 생성
    - Step 2) **2-pass 구조**로 초안을 검토·수정 (한 경로를 더 정교화)

  - Reference

    - *Deliberation Networks: Sequence Generation Beyond One-Pass Decoding* (https://papers.nips.cc/paper_files/paper/2017/file/c6036a69be21cb660499b75718a3ef24-Paper.pdf)

      - NMT에서 제안된 2-pass 디코딩, 이후 LLM reasoning에도 적용
        ```
        In this work, we introduce the deliberation process into the encoder-decoder framework and propose deliberation networks for sequence generation. A deliberation network has two levels of decoders, where the f irst-pass decoder generates a raw sequence and the second-pass decoder polishes and refines the raw sentence with deliberation. Since the second-pass deliberation decoder has global information about what the sequence to be generated might be, it has the potential to generate a better sequence by looking into future words in the raw sentence.
        ```

    - *Think Twice: Deliberation Improves Large Language Models* (https://arxiv.org/pdf/2311.10227)

      - LLM이 **“다시 생각하기” 루프**를 거치면 정밀한 추론 성능이 향상

![figure2](/assets/img/llm/img809.png)

![figure2](/assets/img/llm/img810.png)

<br>

- **Reflection / Self-Consistency in Depth**

  - Procedure

    - Step 1) 모델이 답안을 낸 뒤 **자기 피드백(Reflection)**을 생성
    - Step 2) 피드백을 반영해 다시 수정·보강 (깊이를 늘려 정답 안정화)

  - Reference

    - *Reflexion: Language Agents with Verbal Reinforcement Learning* (https://arxiv.org/pdf/2303.11366)

      - 에이전트가 답을 낸 뒤 자기 피드백을 통해 **점진적으로 답변 품질을 개선**

      ```
      We propose Reflexion, a novel framework to reinforce language agents not by updating weights, but instead through linguistic feedback. Concretely, Reflexion agents verbally reflect on task feedback signals, then maintain their own reflective text in an episodic memory buffer to induce better decision-making in subsequent trials. 
      ```

    - *Self-Refine: Iterative Refinement with Self-Feedback* (https://arxiv.org/pdf/2303.17651)

      - 후보 답안을 생성한 뒤 모델 스스로 피드백을 주고, 그에 따라 **수정/재랭킹**하는 구조
      - (Breadth에도 해당되지만) 반복적 자기수정은 **한 경로를 점점 깊게 발전**시키는 형태도 가능

![figure2](/assets/img/llm/img805.png)

<br>

- **Budget Forcing / Test-time Scaling (Depth Control)**
  - Procedure
    - Step 1) 추론 시 **토큰 예산**(budget)을 강제로 크게 부여
    - Step 2) 모델이 **더 길게 생각**하도록 유도 → 성능 우상향 곡선 확보
  - Reference
    - *s1: Simple test-time scaling* (https://arxiv.org/pdf/2501.19393)
      - 단순 SFT(1K 데이터) + **budget forcing**만으로도 o1류 모델처럼 **더 깊게 생각할수록 성능이 오르는 곡선** 재현

![figure2](/assets/img/llm/img811.png)

<br>

### c) **Search(탐색 구조화)** – 트리/그래프 탐색

- **Tree-of-Thought**:
  - 중간 생각을 분기해 **여러 inference 나무**를 탐색
  - **합의/재귀적 수정**으로 품질을 높임. 
- **Forest-of-Thought**: 
  - 희소 활성화·동적 자기수정·합의 유도를 통해 효율/정확도 개선을 보고. 

<br>

### d) **Adaptive Budget(적응적 예산)** – 난이도에 따라 다르게

**동적 스케일링**

- "쉬운" 문제는 샘플/토큰을 "적게"
- "어려운" 문제는 샘플/토큰을 "많게"

<br>

## (4) Examples

**예시 A: Self-Consistency (다수결)**

- **상황**: 수학 단답형.

- **방법**: 같은 프롬프트를 temperature>0로 $N$회 생성 

  → 정답 후보들 중 최빈값 채택.

```python
def self_consistent_solve(llm, prompt, n=10, temp=0.7):
    answers = []
    for _ in range(n):
        ans = llm(prompt, temperature=temp)
        answers.append(extract_final_answer(ans))
    return majority_vote(answers)
```

<br>

**예시 B: Best-of-N + 재랭킹(검증기 결합)**

- **상황**: 코드 문제.
- **방법**: 후보 코드 N개 생성 → **test케이스 실행/정적 분석/LLM-judge** 점수로 1위 선택.

```python
cands = [gen_code(prompt) for _ in range(N)]
scores = [unit_test_score(code) for code in cands]  # 또는 LLM-judge 점수
best = cands[np.argmax(scores)]
```

<br>

**예시 C: Forest-of-Thought(탐색 + 합의)**

- **상황**: 단계적 "논증"이 필요한 "복잡한 문제"
- **방법**: 여러 inference 나무를 병렬로 확장 → 중간에 **합의/자기수정** → 최종 정리.
- **포인트**: 탐색을 구조화해 단일 체인 한계를 보완(성능↑, 다만 구현 복잡/비용↑). 

<br>





## **예시 D: o1류 “inference 토큰” 증액**





- **상황**: “답하기 전 더 오래 생각” 옵션이 있는 model(o1 등).
- **방법**: 난도가 높은 질문에서 **inference 노력(effort)** 을 올려 더 많은 내부 토큰/단계를 사용.
- **효과**: 정확도↑(특히 난문), 지연/비용↑ → **적응적**으로만 사용 권장. 





------





# **5) 모범 사용법 & 팁**





- **예산 관리**: 총 **샘플 수/탐색 폭/토큰 길이**에 상한을 두고, 문제 난이도 추정으로 **adaptive** 운용. 
- **검증기 결합**: 규칙 기반 채점, test케이스, 외부 툴(계산기/파서), LLM-judge 등으로 **후처리 재랭킹**.
- **캐싱/중간 산출 저장**: 동일/유사 질문에 재사용(지연·비용 절감).
- **재현성**: seed·스냅샷 고정, 로그 남기기(다중 샘플 비결정성 관리).
- **안전/윤리**: 더 많은 샘플이 **유해 출력 위험**도 늘릴 수 있으니, 필터/가드레일을 함께 스케일.





------





# **6) 한계·주의**





- **비용/지연 증가**: 무작정 N을 키우면 한계효용↓, 응답시간↑. **동적 스케일링** 필수. 
- **평가의 공정성**: 같은 model이라도 TTS 예산(샘플 수/토큰 수)을 다르게 주면 성능이 달라 **공정 비교가 어려움**.
- **설계 복잡성**: 탐색·표결·검증 파이프라인이 복잡해지고, 구현 품질에 성패 좌우.
- **TTA와의 혼동 금지**: TTS는 **inference compute 확장**, TTA는 **입력 변환 앙상블**입니다. 





------





## **한 줄 요약**





**Test-time scaling = inference 시 연산을 더 쓰는 전략**입니다.

N-샘플 표결, 트리 탐색, 재랭킹, inference 토큰 증액 등으로 **정확도**를 올리되, **적응적 예산**으로 **비용·지연**을 관리하는 것이 핵심입니다. 



필요하시면, **당장 적용 가능한 코드 스니펫**(예: self-consistency + LLM-judge 재랭킹 + 예산 스케줄링)을 만들어 드릴게요.