---
title: (OpenAI API) Structured Output
categories: [LLM]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# (OpenAI API) Structured Output

<br>

## 1. Structured Output이란?

모델이 반드시 **“정해진 JSON Schema에 맞는 구조화된 데이터”**만 출력하도록 강제하는 OpenAI API 기능

- 즉, 모델은 자연어를 생성하지 않고,
- **JSON Schema 기반 “구조화된 데이터 생성 엔진”처럼 행동**

<br>

## 2. Structured Output의 목표

“LLM이 토큰을 생성하더라도, 최종 출력이 **정확한 JSON 구조**로만 나오게 하고 싶다!”

<br>

아래의 문제들이 해결됨

- JSON이 깨짐
- 큰따옴표 빠짐
- key 누락
- 타입 불일치
- enum 값 잘못됨
- 중괄호 안 닫힘
- 문자열이 자연어로 섞여 나옴
- hallucination으로 구조 틀림

<br>

## 3. 핵심: 모델의 디코딩(decoding mode)을 바꾼다

Structured Output에서 *가장 중요한 점*:

= 모델이 “자연어 생성” 모드가 아니라 “구조화된 JSON 생성” 모드로 작동한다!

<br>

즉, **Sampling/Decoding 단계 자체에서 JSON schema validation이 적용**

<br>

## 4. 작동 방식

모델에, 아래의 정보(=지켜야할 출력 양식)도 함께 보내줘야함!

```python
response_format: {
  "type": "json_schema",
  "schema": { ... JSON Schema ... }
}
```

<br>

이걸 설정하면 모델은 **Schema 위반 시 output 불가**

$$\rightarrow$$ 내부적으로 sampling 시 schema constraint가 걸려서 구조를 절대 어기지 않음

<br>

이게 **“schema-guided decoding”** 방식!

- 모델이 output을 생성하는 동안 schema validator가 작동

<br>

## 5. 예시

Schema:

```python
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "number"}
  },
  "required": ["name", "age"]
}
```

<br>

모델 답변 (항상 완벽):

```python
{
  "name": "Alice",
  "age": 25
}
```

<br>

```python
from openai import OpenAI
client = OpenAI()

completion = client.responses.create(
    model="gpt-4.1-mini",
    input="Extract name and age.",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "Person",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"}
                },
                "required": ["name", "age"],
                "additionalProperties": False
            }
        }
    }
)

print(completion.output[0].content[0].text)
```

