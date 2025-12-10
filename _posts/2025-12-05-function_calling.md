---
title: Function Calling
categories: [LLM]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Function Calling

## 1. 개념

LLM이 **"구조화된 방식"**으로 **"외부 함수(코드)"**를 호출하도록 만드는 기능

== LLM에게 **“자연어 → 함수 호출(JSON)” 변환 능력**을 주는 것

== **“모델이 스스로 적절한 함수를 선택"**해 올바른 파라미터로 호출하도록 하는 기능

<br>

한 줄 요약 = **LLM이 함수 호출 형태(JSON)를 자동 생성해서**, **외부 코드·API·DB와 연결되도록 해주는 기능.**

<br>

## 2. 필요성

LLM은 자연어 처리에는 강하지만....

- 계산(예: 날짜 계산, 복잡한 수학)
- 데이터베이스 조회
- API 호출
- 특정 비즈니스 로직 실행

같은 **정확한 작업은 직접 수행할 수 없음**

<br>

$$\rightarrow \therefore$$ **LLM이 “함수 호출을 구조적으로 만들어주는 역할”**을 하게 함 +  실제 실행은 다른 코드가!

<br>

## 2. Function Calling이 하는 일

유저 입력: *“내일 서울 날씨 알려줘”*

LLM은 자연어로 답변하는 대신...

```
{
  "name": "getWeather",
  "arguments": {
    "location": "Seoul",
    "date": "2025-03-19"
  }
}
```

처럼 **정확한 함수 호출 JSON**을 만들어서 반환.

<br>

함수 스펙 (후보군)

- 개발자가 API에 “함수 정의(schema)”로 사전 제공함)

<br>

여기서:

- 어떤 함수를 호출할지 **LLM이 선택**
- 함수 인자들을 **정확한 JSON으로 구성**

<br>

## 3. 실제 시스템 흐름

### Step 1) 개발자가 함수들을 정의함

```
{
  "name": "getWeather",
  "description": "특정 날짜의 날씨를 가져오는 함수",
  "parameters": {
    "type": "object",
    "properties": {
      "location": { "type": "string" },
      "date": { "type": "string" }
    },
    "required": ["location", "date"]
  }
}
```

<br>

### Step 2) 유저 질문 → 모델 호출

모델은 자연어를 이해하고 **함수 호출 JSON**으로 응답.

<br>

### Step 3) 개발자 코드가 그 JSON을 받아 함수 실행

(ex. 외부 기상 API 호출)

- **실행 결과를 다시 모델에 전달**하면, **LLM이 최종 문장으로 포맷팅**해준다.

<br>

## 4. Function Calling의 장점

- 자연어 → 정형화된 API 호출 자동 변환
  - LLM이 **“적절한 함수/적절한 파라미터”를 자동으로** 선택

- 강력한 툴 사용 가능
  - LLM이 직접 계산할 필요 없이
  - **코드/DB/API 호출**을 사용

- 오탈자·형식 오류 없음
  - LLM이 JSON schema를 보고 **정확히 맞춘 형식**으로 출력

- 안전하고 deterministic한 통합
  - 자연어 hallucination 없이 **함수 호출**만 하게 만들 수 있음.

<br>

## 5. 비유

- **LLM은 “똑똑한 비서”이고,**

- **function calling은 “비서가 정보를 구조화해서 업무 요청을 만드는” 과정.**

<br>

비서는 직접 날씨를 아는 게 아니라,

→ 날씨 API를 호출하는 “요청서(JSON)”를 만들어주고

→ 개발자가 그 요청을 실제로 처리하는 것.

