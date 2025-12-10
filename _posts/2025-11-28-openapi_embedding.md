---
title: (OpenAI API) OpenAPI Embedding
categories: [LLM]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# (OpenAI API) OpenAPI Embedding

<br>

## 1. Models

`text-embedding-3-small`

- 가성비 좋고 속도 빠른 기본 모델
- 대부분의 검색/분류/추천에는 이걸 먼저 쓴다고 보면 됨.

<br>

`text-embedding-3-large`

- 더 높은 성능, 다국어 이해력 강화
- 랭킹/고품질 검색 등에 적합

<br>

## 2. Embeddings 기본 호출 방식

```python
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=[
        "Hello, world!",
        "This is another sentence."
    ],
)

emb1 = response.data[0].embedding   # 리스트[float]
emb2 = response.data[1].embedding

print(len(emb1))  # 보통 1536 등 차원 수
```

- input에 문자열 하나 or 여러 개 전달
- `response.data[i].embedding`이 바로 벡터 (리스트[float])
- 이 벡터를 **벡터DB, Faiss, Annoy, Elastic, Postgres pgvector** 등에 저장해서 사용

<br>