---
title: JSON vs JSONL
categories: [PYTHON, LLM]
tags: []
excerpt: 

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# JSON vs. JSONL

<br>

# 1. JSON

구조가 **tree형 (배열 + 객체)**

```python
{
  "users": [
    {"id": 1, "name": "Tom"},
    {"id": 2, "name": "Amy"}
  ]
}
```

- 장점: 구조가 명확함
- 단점: ***한 번에 전체를 읽어야 함 (대량 데이터에 불리함)***

<br>

# 2. JSONL

줄(line) 단위로 **하나의 JSON 객체**

```python
{"id": 1, "name": "Tom"}
{"id": 2, "name": "Amy"}
```

- 장점:
  - 스트리밍 처리에 좋음
  - 매우 큰 데이터셋도 한 줄씩 처리 가능
  - ML 학습 데이터(특히 OpenAI fine-tuning)에서 표준 형식
- 단점: 전체 구조(배열 형태)를 강제하지 않음

<br>

## 3. Summary

- **JSON** = 하나의 큰 구조
- **JSONL** = 줄마다 하나의 JSON (많은 데이터에 적합)

<br>

| **형식**               | **특징**                                                     | **언제 사용?**                                      |
| ---------------------- | ------------------------------------------------------------ | --------------------------------------------------- |
| **JSON**               | 하나의 **전체 문서**가 JSON 구조(객체/배열)                  | 완전한 데이터 구조를 한 번에 주고받을 때            |
| **JSONL (JSON Lines)** | **각 줄마다 하나의 JSON 객체** <br />→ 여러 줄 = 여러 개의 레코드 | 대규모 데이터 처리, 스트리밍, 대량 샘플 학습 데이터 |





