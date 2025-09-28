# 1. Agent의 정의와 철학적 기초

## (1) Agent란?

**환경**과 **상호작용**하며, 주어진 **입력**을 바탕으로 **행동**을 결정하는 **의사결정** 주체

$\text{Agent} : \text{Perception} \rightarrow \text{Action}$.

- Input = 환경으로부터의 관측 (**Observation, State**)
- Output = 행동 (**Action**)

<br>

## (2) Agent의 4가지 구성 요소

| **구성 요소**         | **설명**                                |
| --------------------- | --------------------------------------- |
| 1. **Sensor (입력)**  | 환경으로부터 얻어지는 정보              |
| 2. **Perceptor**      | 얻어진 (관측된) 정보를 내부 상태로 변환 |
| 3. **Policy/Planner** | 어떤 행동을 취할지 결정                 |
| 4. **Actuator**       | 실제 행동을 수행함                      |

<bR>

## (3) 일반 ML 모델 vs. Agent

| **항목** | **일반 ML 모델**       | **AI Agent**                          |
| -------- | ---------------------- | ------------------------------------- |
| Input    | 고정된 dataset         | **(실시간) 환경 관측 (sensor input)** |
| Output   | 예측값                 | **행동 (action)**                     |
| 목표     | 정답 예측 (e.g., 분류) | 목표 달성, 보상 최적화                |
| 상호작용 | X                      | O                                     |
| 시간 축  | Static (batch)         | Dynamic (time-series)                 |

<br>

### Example

일반 ML 모델

- **Input: 고양이 사진**
- **Output: “고양이”**
- 더 이상 아무 행동 없음 ... 예측이 전부!

Agent

- **Input: 로봇 센서 (앞에 장애물 있음)**
- **Output: 왼쪽으로 15도 회전**
- 환경이 바뀜 → 다시 입력 → 반복 → 목표 도달

<br>

# 2. Agent의 분류

구조 & 설계 방식 에 따라 나뉨!

- (1) 구조
- (2) 

<br>

## (1) 구조 (Architecture)

### a) Reactive (반응형) Agent 

- (기억이나 내부 상태 없이) **즉각적인 반응**에만 수행!
- **단순 조건 기반**으로 설계 (e.g., `if x then y`)

<br>

Example)

```python
if obstacle_in_front:
    turn_left()
```

<br>

Pros & Cons

- Pros) 빠르고 가벼움
- Cons) 복잡한 환경에는 적합 X

<br>

### b) Model-based / Goal-based Agent

- (Reactive agent과 달리) **내부 모델** 존재 $\rightarrow$ 환경 상태 추론
- **목표 (goal)**를 고려하여 **계획(plan)**을 세움! $\rightarrow$ 이를 통해 행동을 선택함

<br>

Example) 

- Chess 에이전트: 다음 수를 시뮬레이션하여 **승률이 높은 수**를 선택

<br>

Pros & Cons

- Pros) 복잡한 환경에 적합 O
- Cons) 높은 연산량

<br>

### c) Utility-based Agent

- 목표 이상으로, **“얼마나 좋은지”**를 측정하는 **Utility Function** 존재
- 단순히 목표 달성 외에도 최적/최선의 선택을 하도록 설계됨

<br>

Example) 자율주행차가 **“충돌 회피 + 연료 효율 + 승차감”**을 동시에 고려

- 목표: 자율 주행
- Utility: 그 외의 것들

<br>

### d) Learning (학습형) Agent 

- **환경과의 상호작용**을 통해 **경험 기반** 학습
- 보상을 통해 최적의 policy를 학습함 (e.g., RL)

<br>

## (2) 학습 방식에 따른 분류

| **유형**          | **설명**                      | **예시**                |
| ----------------- | ----------------------------- | ----------------------- |
| a) Rule-based     | 사람이 정한 규칙 기반 행동    | expert system, 게임 NPC |
| b) Planning-based | 목표까지의 경로 탐색          | STRIPS, GOAP, A*        |
| c) RL Agent       | 보상 기반 학습                | DQN, PPO, AlphaZero     |
| d) LLM Agent      | 언어를 통해 추론 및 도구 사용 | ReAct, AutoGPT, BabyAGI |

<br>

```
Agent 유형
├── Reactive (반응형)
├── Model-based (모델 기반)
├── Utility-based (효용 기반)
├── Learning Agent
     ├── RL 기반
     └── LLM 기반
```

<br>

# 3. Agent Architectures)

Agent의 내부 구성 & 동작 방식을 결정

<br>

## 대표적인 Architecture

| **아키텍처**                    | **개념**       | **설명**                                                   |
| ------------------------------- | -------------- | ---------------------------------------------------------- |
| 1. **Simple Reflex** Agent      | 조건-행동 규칙 | **if 조건 → 행동 (메모리 없음)**                           |
| 2. **Model-based Reflex** Agent | 환경 모델 포함 | **내부 상태**를 업데이트하며 동작                          |
| 3. **Goal-based** Agent         | 목표 고려      | 단순 반응이 아닌, **“목표 달성”** 중심                     |
| 4. **Utility-based** Agent      | 효과 측정      | 목표 달성 뿐 아니라 **“좋은 행동 (효용까지 고려)”**을 고름 |
| 5. **Learning** Agent           | 경험 기반 갱신 | 행동 전략을 **경험을 통해** 계속 개선함                    |

<br>

## (1) Simple Reflex ~

한 줄 요약: **입력에 즉각 반응**하는 규칙 기반 시스템 (모델, 메모리 X)

```python
if obstacle_ahead:
    turn_left()
```

Pros & Cons

- Pros: 빠르고 단순
- Cons: 복잡한 환경 대응 불가

<br>

## (2) Model-based Reflex ~

한 줄 요약: 내부적으로 **상태(state)**를 유지하여, 이전 상태나 **환경 모델**을 고려하여 반응

- Perception을 통해 다음 상태 추론

<br>

Example) 자율주행차

```python
state = update_state(perception, previous_state)

if state == "about_to_crash":
    brake()
```

- `previous_state` → **이전 시점까지 알고 있던 내부 상태**
  - 예: 이전 위치, 이전 속도, 이전 방향
- `perception` → **현재 센서로부터 들어온 관측 정보**
  - 예: 레이더 거리, 카메라 영상, 장애물 여부 등
- `state` → **지금 시점에서 추론된 새로운 상태**
  - 즉, state = f(previous_state, perception)으로 업데이트된 정보

<br>

## (3) Goal-based ~

한 줄 요약: (단순 반응 X) “**목표 달성**”을 위한 계획 수행!

- **목표(goal)**를 설정 → 이를 달성하기 위해, **경로(plan)**를 찾아 행동

<br>

Example) 자율주행차

- 목표: **“목적지 A 도달”**
- **A\* 알고리즘**으로 경로 계획하기

<br>

## (4) Utility-based ~

한 줄 요약: 가장 나은 ( = 효용 극대화 ) 하면서 목표 달성!

- 목표 & 효용 (utility)를 모두 고려함
- **효용 함수 (utility function)**를 통해 행동 quality 평가

<br>

Example) 자율주행차

-  같은 목적지라도 “빠름”, “연료 적음”, “안전” 중 가장 유리한 선택

<br>

## (5) Learning ~

한 줄 요약: 환경과의 상호작용을 통해 **행동 전략을 "학습"**

<br>

Components

- **학습 모듈**: policy 업데이트
- **비평가 (critic)**: 현재 전략 평가
- **행동자 (actor)**: 실제 행동 수행

<br>

Example) (RL algorithm) DQN, PPO 등

<br>

## Comparison

단순 -> 복잡

```
Reflex → Model-based → Goal-based → Utility-based → Learning
```

<br>

# 4. Perception – Decision – Action loop

- Agent가 환경과 상호작용하는 기본 동작 흐름
- 모든 Agent의 기본 구조를 구성함

<br>

## 정의

- a. AI Agent는 지속적으로 **환경을 관찰(perception)**하고,

- b. 이를 기반으로 **판단(decision)**을 내리며,

- c. **행동(action)**을 통해 환경에 영향을 미침

$\rightarrow$ 이 모든 것이 **하나의 루프(loop)**로 반복!

<br>

## (1) Perception (관측)

환경으로부터 **센서를 통해 정보를 받아옴**

- ex)  눈, 마이크, 위치 센서, API 입력 등
- 참고: 입력은 완전할 수도, 불완전할 수도 있음!

<br>

Example)

```python
observation = get_sensor_input()
```

<br>

## (2) Decision (판단)

관측된 정보를 바탕으로 **내부 상태(state)를 갱신** &  (policy나 planning 알고리즘에 따라) **행동을 선택**

<br>

Example)

```python
state = update_state(observation, previous_state)
action = policy(state)
```

- policy: $\pi(s) \rightarrow a$
- planning: $\text{plan}(\text{goal}, \text{current\_map}) \rightarrow \text{action sequence}$

## (3) Action (행동)

선택된 행동을 환경에 **적용**

$\rightarrow$ 이에 따라 환경이 변화 & 다음 loop의 입력이 바뀜

<br>

Example) 

```python
execute_action(action)
```

<br>

## Summary

```python
state = initial_state()

while True:
    # (1) Perception
    observation = get_observation()
    
    # (2) Decision
    state = update_state(state, observation)
    action = policy(state)
    
    # (3) Action
    execute_action(action)
```

<br>

### Example 1: 자율주행차

| **단계**   | **실제 시스템 구성**                  |
| ---------- | ------------------------------------- |
| Perception | 카메라, LiDAR, GPS 센서 입력          |
| Decision   | (현재 위치와 속도 기반으로) 경로 계획 |
| Action     | 스티어링, 브레이크, 가속 조작         |

<br>

### Example 2: LLM 기반 Agent

| **단계**   | **예시**                           |
| ---------- | ---------------------------------- |
| Perception | Input prompt                       |
| Decision   | LLM → CoT, 툴 선택, 함수 호출 결정 |
| Action     | API 호출, 툴 사용, 답변 생성       |

<br>

### 기타 Extension) POMDP

**Partially Observable Markov Decision Process**

- “환경의 상태를 완전히 관측할 수 없는 상황에서의 의사결정 프레임워크”

- **MDP에 관측 제약이 추가된 것**

  (즉, 관측이 불완전할 수 있음을 가정!)

<br>

POMDP + Agent

- 내부적으로 **belief state**를 유지!
- 단순한 상태가 아닌 **확률 분포**를 바탕으로 행동!

<br>

# 5. LearningAgent (RL 기반)

Agent가 환경과의 상호작용을 통해 **스스로 행동 전략을 학습**하는 구조

$\rightarrow$ 환경에서의 경험을 통해 어떤 행동을 해야 가장 이득(보상)이 되는지 학습!

<br>

## (1) Components

| **구성 요소** | **기호**           | **설명**                                |
| ------------- | ------------------ | --------------------------------------- |
| 상태          | $s \in S$          | 환경의 현재 상태                        |
| 행동          | $a \in A$          | agent가 선택할 수 있는 행동             |
| 보상          | $r \in \mathbb{R}$ | 한 행동 이후 받는 즉시 보상             |
| 정책          | $\pi(a \mid s)$    | 상태에서 어떤 행동을 할지 결정하는 함수 |
| 가치 함수     | $V(s), Q(s,a)$     | 미래 보상의 기대값                      |
| 환경          | $T(s’ \mid s,a)$   | 행동 후 다음 상태로의 전이 확률         |

<br>

## (2) 작동 흐름 (Agent–Environment Loop)

- Step 1) Agent는 현재 상태 $s_t$를 관측
- Step 2) 정책 $\pi$에 따라 행동 $a_t$ 선택
- Step 3) 환경은 보상 $r_t$와 다음 상태 $s_{t+1}$ 반환
- Step 4) Agent는 이를 기억하고 학습에 반영

$\rightarrow$ 위 Loop 반복을 통해 policy 점차 개선

<br>

## (3) RL algorithms

### a) Value-based: Q-learning, DQN

- 핵심 아이디어: $Q(s,a) \leftarrow Q(s,a) + \alpha (r + \gamma \max_a Q(s’,a) - Q(s,a))$.
- 어떤 행동이 **가치 있는지**를 테이블 또는 NN으로 추정

<br>

### b) Policy-based: REINFORCE, PPO

- 정책 $\pi(a \mid s)$ 자체를 확률적으로 직접 학습

- Gradient ascent:

  $\nabla J(\theta) = \mathbb{E}[ \nabla_\theta \log \pi_\theta(a \mid s) R ]$.

<br>

### c) Actor–Critic 계열

- Policy + Value를 동시에 학습
- 대표: A2C, PPO, DDPG

<br>

# 6. Multi-agent Systems (MAS)

## (1) 정의

(여러 개의 agent가 상호작용하는 시스템)

- 여러 개의 agent가 동시에 존재
- 공동 환경** 내에서 **서로 영향을 주고받는** 구조

<br>

특징

- agent들 간에는 **경쟁적**, **협력적**, 또는 **혼합된 관계**가 존재
- 각 agent는 ..
  - 독립적인 정책과 목표를 가질 수도 있고
  - 공통 목표를 향할 수도 있음

<br>

## (2) 유형

| **유형**              | **설명**                          |
| --------------------- | --------------------------------- |
| **Fully Cooperative** | 모든 agent가 **같은 목표**        |
| **Fully Competitive** | 각 agent가 **독립적/상반된 목표** |
| **Mixed**             | 일부는 **협력**, 일부는 **경쟁**  |

<br>

## (3) 핵심 과제

| **과제**              | **설명**                                        |
| --------------------- | ----------------------------------------------- |
| **Communication**     | agent 간 정보 공유는 언제/어떻게?               |
| **Credit Assignment** | (공동 보상 시) 어떤 agent가 기여?               |
| **Non-stationarity**  | 다른 agent의 정책이 계속 바뀜 ->  학습이 어려움 |
| **Scalability**       | agent 수가 늘어나면 -> complexity issue         |
| **Emergent behavior** | 예기치 않은 집단 행동                           |

<br>

# 7. LLM-based AI Agent

한 줄 요약: LLM을 두뇌로 해서, agent처럼 사고하고 행동하는 시스템

<br>

## (1) 정의

LLM 기반 Agent는 **GPT-4, Claude, Gemini 등 LLM**을 기반으로,

인간처럼 **계획 + 도구를 사용 + 추론 + 행동** Agent

<br>

## (2) 핵심 구성요소

| **구성 요소**   | **설명**                                        |
| --------------- | ----------------------------------------------- |
| LLM Core        | “두뇌” 역할: Text 기반 추론                     |
| Memory          | 과거의 대화, 상황, 행동 로그 저장               |
| Tools / Actions | e.g., 외부 API 호출, 코드 실행, 검색, 계산기 등 |
| Planner         | 고수준 Goal -> 여러 개의 저수준 Goal로 세분화   |
| Controller      | "LLM ↔ 환경" 상호작용 반복 수행 (loop)          |

<br>

## (3) 대표적 알고리즘 (Prompt 및 실행 구조)**

### a) ReAct (Reasoning + Acting)

한 줄 요약: **Thought & Action**을 번갈아 수행하며 문제 해결

<br>

**Prompt 형태**

```
Question: What is the capital of the country where the Eiffel Tower is?
Thought 1: Eiffel Tower is in Paris. I need to find the country.
Action 1: search("Eiffel Tower country")
Observation 1: France
Thought 2: The capital of France is Paris.
Answer: Paris
```

특징

- **step-by-step** 추론 + 도구 사용
- 구조화된 format으로 LLM의 reasoning을 외부와 연결

<br>

### b) AutoGPT / BabyAGI

한 줄 요약: **고수준 목표**를 주면, LLM이 스스로 **"하위 작업을 계획"**하고 실행하는 Agent!

<br>

특징

- **자체적인 “Task Queue”를 운영**
- **“Create → Execute → Reflect”** loop 구조
- 외부 도구, 메모리, 파일 시스템과 통합

<br>

Pros & Cons

- Pros) **"자동화"** 
- Cons) 실패가 누적될 수도!

<br>

### c) CoT + Tool use

한 줄 요약: CoT를 유도 + 특정 단계에서는 Tool 호출함으로써 보완하기

<br>

Prompt 예시:

```python
Thought: First, I need to compute 97 * 23.
Action: calculator("97 * 23")
Observation: 2231
Thought: Now I can proceed to compare with 2000...
```

<br>

### d) Function Calling / Toolformer-style

한 줄 요약: LLM에게 구조화된 **"함수 목록"**을 주고, 응답에서 (JSON 형태로) **"어떤 함수를 언제 호출할지 판단"**하게 함!

<br>

Prompt 예시:

```python
{
  "function": "search",
  "args": {"query": "population of Seoul"}
}
```



- GPT-4 Function Calling 또는 OpenAI function tool 사용
- LangChain, LlamaIndex 등에서 잘 지원





------





## **✅ 7.4 실행 흐름 요약**



```
[User Input]
     ↓
[LLM Prompt + Context]
     ↓
[LLM Output: Action / Thought / Answer]
     ↓
[Tool 사용 (API, 계산기, 코드 실행 등)]
     ↓
[Observation → LLM에 재입력]
     ↓
(반복)
```



------





## **✅ 7.5 메모리 구조**



| **유형**   | **예시**                              |
| ---------- | ------------------------------------- |
| Short-term | 직전 입력/출력 (1~2 step context)     |
| Long-term  | 요약된 과거 기록, embedding 기반 검색 |
| Episodic   | 특정 상황 전체 저장                   |
| Semantic   | 추상적 지식 형태로 저장               |



------





## **✅ 7.6 사용 프레임워크 & 플랫폼**



| **플랫폼**              | **설명**                                 |
| ----------------------- | ---------------------------------------- |
| LangChain               | 모듈형 에이전트 구조 구현                |
| OpenAI Function Calling | JSON 기반 도구 호출                      |
| LlamaIndex              | 문서 검색 + context 추출                 |
| AutoGen (Microsoft)     | Multi-agent LLM orchestration            |
| CrewAI                  | 다중 LLM 역할 분담 기반 워크플로우       |
| AgentVerse, MetaGPT     | 다양한 에이전트 orchestration 프레임워크 |



------





## **✅ 7.7 주요 연구 과제**



| **분야**           | **내용**                                       |
| ------------------ | ---------------------------------------------- |
| Tool Planning      | 어느 도구를 언제 사용할 것인가?                |
| Memory Compression | 장기 기억을 어떻게 유지할 것인가?              |
| Self-reflection    | LLM이 스스로 오류를 인식하고 수정할 수 있는가? |
| Multi-agent 협력   | 여러 LLM이 역할 분담하며 협업 가능할까?        |
| 안전성과 해석력    | hallucination, 반복 루프 방지 등               |



------





## **✅ 요약 정리**



| **요소**  | **설명**                               |
| --------- | -------------------------------------- |
| LLM Core  | 추론과 텍스트 생성                     |
| Memory    | 대화 기억 또는 임베딩 기반 검색        |
| Tool 사용 | API, 코드 실행 등 외부 기능 연동       |
| 전략      | ReAct, CoT+Tool, AutoGPT               |
| 실용화    | LangChain, OpenAI API, AutoGen 등 활용 |



------



좋습니다!

이제 LLM 기반 에이전트 전략 중 가장 대표적인 두 가지를 **차례대로, 매우 자세히, 예시와 함께** 설명드리겠습니다:



------





# **🧠 7.7.1 ReAct (Reasoning + Acting)**





------





## **✅ 개요**





**ReAct**는 LLM이 문제를 해결할 때,



- **“생각(Reasoning)”**과

- **“행동(Acting)”**을 **교대로 반복**하며

  외부 도구를 사용해 능동적으로 추론과 실행을 수행하도록 하는 **프롬프트 전략 + 에이전트 구조**입니다.





> 처음 제안: [Yao et al., 2022, “ReAct: Synergizing Reasoning and Acting in Language Models”](https://arxiv.org/abs/2210.03629)



------





## **✅ 핵심 아이디어**





> 기존 LLM은 텍스트만 생성하거나,

> 단순한 함수 호출만 가능했지만,

> **ReAct는 Thought와 Action을 번갈아 생성**하여 문제 해결 능력을 강화합니다.



------





## **✅ ReAct 구성 요소**



| **요소**         | **설명**                                             |
| ---------------- | ---------------------------------------------------- |
| **Thought**      | 내부 추론, 계획, 다음 행동을 결정하기 위한 reasoning |
| **Action**       | 도구 호출, 웹 검색, 계산기 사용 등 실제 행동 수행    |
| **Observation**  | Action의 실행 결과                                   |
| **Final Answer** | 문제에 대한 결론                                     |



------





## **✅ 동작 예시**





> Q: Where is the Eiffel Tower located, and what is the population of that country?

```
Thought: The Eiffel Tower is in Paris. I need to find which country Paris is in.
Action: search("Which country is Paris in?")
Observation: Paris is the capital of France.
Thought: Now I need to find the population of France.
Action: search("Population of France")
Observation: France has a population of about 67 million.
Answer: France has a population of about 67 million.
```

✔️ LLM은 **중간 판단을 하고**,

✔️ **외부 도구를 호출한 결과를 기반으로 다음 추론을 진행**합니다.



------





## **✅ ReAct 프롬프트 구조**





ReAct는 특별한 API가 없어도, **프롬프트만으로 구현 가능**합니다:

```
Question: ...
Thought: ...
Action: ...
Observation: ...
Thought: ...
Action: ...
...
Answer: ...
```



------





## **✅ 장점**





- 🧠 **추론과 실행을 결합**하여 더 강력한 문제 해결 가능
- 🔍 Tool 사용이 자연스러움 (웹검색, 계산, 코드 실행 등)
- 🔁 반복 구조를 통해 복잡한 문제 해결 가능





------





## **✅ 단점**





- 🔁 무한 루프 가능성 (Thought → Action이 반복되다가 끝나지 않음)
- 📏 멀티턴 제어가 필요 → Controller 구성 필요
- ❌ OpenAI function calling 같은 structured API와는 다소 다름 (prompt 기반)





------





## **✅ 구현 라이브러리 예**





- **LangChain ReAct agent**
- OpenAI API + Python loop 기반으로 쉽게 구현 가능





------





## **✅ 요약**



| **항목**      | **설명**                     |
| ------------- | ---------------------------- |
| 전략          | Reasoning + Acting 반복      |
| 핵심 단어     | Thought, Action, Observation |
| 프롬프트 기반 | ✅                            |
| 외부 도구     | 검색, 계산, DB 등            |
| 용도          | 복잡한 multi-hop 질의 처리   |



------



이제 다음으로 **AutoGPT**를 아주 자세히 설명드릴게요.

계속 보실까요? (“다음” 또는 “AutoGPT 설명해줘”라고 해 주세요!)



좋습니다!

이제 이어서 설명드릴 LLM 기반 에이전트 전략은 **AutoGPT**입니다.

ReAct보다 더 **자율적이며 장기 목표를 스스로 달성**하려는 구조입니다.



------





# **🤖 7.7.2 AutoGPT**





------





## **✅ 개요**





**AutoGPT**는 LLM이 **스스로 하위 목표(sub-tasks)를 생성**,

그에 따라 행동하며 **외부 세계와 연속적으로 상호작용**하는 **완전 자율형 에이전트 프레임워크**입니다.



> 처음 등장: [Significant Gravitas, 2023, GitHub Auto-GPT 프로젝트](https://github.com/Torantulino/Auto-GPT)



------





## **✅ 핵심 철학**





> “하나의 고수준 목표만 주면, LLM이 자체적으로 문제를 세분화하고, 툴을 이용해 스스로 문제를 해결하게 하자.”



즉, 인간이 “이 논문을 요약하고, 관련 연구를 조사해줘”라는 명령을 내리면

AutoGPT는:



1. 요약 →
2. 논문 검색 →
3. 요약 정리 →
4. 문서 저장





…까지 스스로 수행합니다.



------





## **✅ 전체 구조**



| **구성 요소**     | **역할**                                |
| ----------------- | --------------------------------------- |
| 🎯 Goal            | 사용자로부터 주어진 고수준 목표         |
| 📝 Task Manager    | 목표를 하위 작업(task)으로 분할         |
| 🧠 LLM Core        | 작업 수행, 판단, 문장 생성              |
| 📂 Memory          | 과거 작업 내용 저장                     |
| 🧰 Tools           | 브라우저, 파일 저장, 코드 실행, 검색 등 |
| 🔁 Loop Controller | 다음 작업을 결정하고 실행을 반복        |



------





## **✅ 실행 루프 (AutoGPT style)**



```
User goal: "Create a market analysis report for EV batteries in Korea"

1. Plan: Search recent EV battery trends
2. Action: Use web search tool
3. Observation: Found 3 articles...
4. Thought: Now summarize those articles
5. Action: Use summarizer tool
6. Observation: Got 3 summaries
7. Thought: Now draft a report using summary
8. Action: Write to file "report.txt"
...
```



------





## **✅ 특징 vs ReAct**



| **항목**   | **ReAct**             | **AutoGPT**              |
| ---------- | --------------------- | ------------------------ |
| 방식       | 추론 + 행동 수동 반복 | 자율적 작업 생성 및 반복 |
| Task Queue | 없음                  | ✅ 내부적으로 존재        |
| Memory     | optional              | ✅ 강제적                 |
| Goal 설정  | 외부에서 지속 유도    | 한 번만 주면 됨          |
| 활용도     | 단기 질의 응답        | 장기 목표 수행에 적합    |



------





## **✅ 예시 활용 시나리오**







### **🎯 프로젝트 리서치 도우미**





- “리서치 주제 요약 + 관련 논문 조사 + 보고서 초안 작성”







### **🎯 코드 자동 생성기**





- “간단한 웹 크롤러 코드 짜고 결과를 csv로 저장”







### **🎯 데이터 분석 자동화**





- “데이터 다운로드 → 전처리 → 분석 → 시각화 → 리포트 저장”





------





## **✅ 한계점**



| **한계**           | **설명**                                |
| ------------------ | --------------------------------------- |
| 🔁 무한 루프 가능성 | 목표를 모호하게 이해하고 작업을 반복    |
| 🎯 목표 해석 문제   | High-level goal을 잘게 나누는 능력 한계 |
| 📋 Memory 폭주      | 장기 memory 관리가 불안정               |
| 🧠 Planning 품질    | subtask 생성이 비효율적이거나 중복됨    |



------





## **✅ 프레임워크 예시**



| **플랫폼**       | **설명**                                    |
| ---------------- | ------------------------------------------- |
| **AutoGPT**      | 오리지널 GitHub 오픈소스                    |
| **AgentGPT**     | 웹 기반 GUI 버전                            |
| **GodMode**      | 직관적 multi-agent Web UI                   |
| **BabyAGI**      | TaskQueue + LangChain 기반의 간단한 AutoGPT |
| **AutoGen (MS)** | 다중 agent 간 대화 오케스트레이션 지원      |



------





## **✅ 구현 포인트 (AutoGPT 스타일 에이전트 만들기 위해 필요한 것)**



| **요소**    | **구현 방식**                                                |
| ----------- | ------------------------------------------------------------ |
| Goal 입력   | 유저 prompt 또는 task 지정                                   |
| Task 생성기 | LLM이 sub-task를 생성                                        |
| Memory 모듈 | vector store 또는 요약                                       |
| Tool 모듈   | 검색, 계산기, 파일 저장 등                                   |
| 루프 제어   | agent가 스스로 계속 수행하거나 human-in-the-loop로 멈춤 제어 |



------





## **✅ 요약**



| **항목**   | **내용**                                |
| ---------- | --------------------------------------- |
| 전략       | LLM이 subtask를 생성하며 자율 루프 수행 |
| 프롬프트   | Goal → Thought → Action → Observation   |
| 특징       | Planning + Tool + Memory + Loop         |
| 프레임워크 | AutoGPT, BabyAGI, AutoGen 등            |



------



이제 원하시면:



- ReAct vs AutoGPT 비교표
- 실제 AutoGPT 스타일 LLM 프롬프트 예시
- 당신이 만드는 agent 설계 템플릿





도 도와드릴 수 있습니다. 원하시나요?