# 3. 프롬프트 엔지니어링의 첫 번째 단계

## (1) Introduction

2장: LLM을 사용하여, 자연어 쿼리를 통해 관련 문서를 빠르게 찾을 수 있는 "비대칭 의미 기반 검색 시스템"을 구축함.

But, 단순히 검색으로 끝나서는 안됨!

User experience 향상시키기 위해, ***End-to-End LLM 기반***의 application을 생성해야!

$\rightarrow$ Necessity of ***PROMPT ENGINEERING***!!

<br>

## (2) 프롬프트 엔지니어링 (Prompt Engineering)

요약

- 효과적으로 작업을 전달하여,

- 정확하고 유용한 출력을 반환하도록 유도하는

- **LLM에 대한 입력(프롬프트)**를 만드는 것!

한 줄 요약

$\rightarrow$ ***원하는 output을 위해 LLM에 input을 구성하는 방법***

<br>

### a) LLM에서 정렬 (Alignment)

LLM이 어떻게 학습되는지 뿐만 아니라, LLM이 어떻게 사람의 입력에 정렬(alignment)되는지 알아야!

***Alignment?***

- 모델이 사용자가 예상한것과 일치하는 방식으로 입력 프롬프트를 이해하고 답변하는 것!
- if not, 관련 없거나 잘못된 답변 생성!

<br>

최근에는, 몇몇의 LLM은 추가적인 정렬 기능과 함께 개발되었음!

- ex) Anthropic의 RLAIF (Constitutional AI-driven Reinforcement Learning from AI Feedback)
- ex) OpenAI의 GPT 계열: RLHF (Reinforcement Learning from Human Feedback)

$\rightarrow$ 이러한 정렬 기술은, 특정 프롬프트를 이해하고 답변하는 모델의 능력을 향상시킴!

<br>

### b) 직접 요청하기

Prompt engineering의 가장 중요한 규칙: ***"요청하는 내용이 최대한 명확 + 직접적이어야"***

더 명확한 LLM의 답변을 위해, "접두사를 추가"하여 명확하게 표시할 수 있음.

<br>

### c) Few-shot Learning

작업에 대한 깊은 이해가 필요한 복잡한 작업의 경우, 몇 개의 예제를 LLM에 제공해주면 더 도움이 될 것!

그러면, LLM이 일부의 예제를 바탕으로 추론할 수 있게 될 것!

<br>

### d) 출력 구조화

