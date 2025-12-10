---
title: (OpenAI API) OpenAI Fine-tuning
categories: [LLM]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# (OpenAI API) OpenAI Fine-tuning

## 1. 개요

OpenAI fine-tuning

- 사용자: 데이터만 제공
- OpenAI: learning rate, optimizer, loss 등은 OpenAI가 자동 관리

<br>

Updated 모델:

```
ft:gpt-4.1-mini:your-team:xxxx-xx-xx
```

- 위와 같은 형태로 API로 호출해 사용

- 모델 weight는 사용자가 볼 수 없음 & 다운로드도 불가능 (closed-source니까 당연)

  오직 API로 inference만 가능!

<br>

## 2. 기본 코드

https://platform.openai.com/docs/guides/model-optimization

<br>

### Step 1) Training 파일(JSONL) 준비

`training_data.jsonl` 예시:

```python
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi! How can I help you?"}]}

{"messages": [{"role": "user", "content": "Summarize this text."}, {"role": "assistant", "content": "Sure! Here's the summary..."}]}
```

- 1줄 = 1개의 sample
- format = **messages[]** (ChatCompletion 스타일)

<br>

### Step 2) 파일 업로드

```python
from openai import OpenAI

client = OpenAI()

file = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)
```

<br>

### Step 3) Fine-tuning Job 생성

```python
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4.1-mini"   # fine-tuning 가능한 모델 선택
)

print(job)
```

```
id='ftjob-abc123'
status='pending'
model='gpt-4.1-mini'
```

<br>

### Step 4) Job 상태 조회 (Monitoring )

```python
job_status = client.fine_tuning.jobs.retrieve(job.id)
```

<br>

(또는 이벤트 스트림 확인)

```python
for event in client.fine_tuning.jobs.list_events(job.id):
    print(event)
```

<br>

### Step 5) Fine-tuned 모델 사용

학습이 완료되면 **모델 ID**가 생김

```
ft:gpt-4.1-mini:abc123:2025-01-20
```

<br>

사용 방법:

```python
response = client.responses.create(
    model="ft:gpt-4.1-mini:abc123",
    input="Explain quantum computing in simple terms."
)

print(response.output[0].content[0].text)
```

<br>

## 3. 기타

### (1) (Optional) Validation set

```python
validation_file = client.files.create(
    file=open("validation.jsonl", "rb"),
    purpose="fine-tune"
)

job = client.fine_tuning.jobs.create(
    training_file=file.id,
    validation_file=validation_file.id,
    model="gpt-4.1-mini"
)
```

<br>

### (2) Fine-tuned 모델 목록 확인

```python
client.fine_tuning.jobs.list()
```

<br>

## 4. Summary

```
training_data.jsonl 준비
        ↓
client.files.create()
        ↓
client.fine_tuning.jobs.create()
        ↓
모델 학습 (OpenAI 서버)
        ↓
완료 후 모델 ID 생성
        ↓
API 호출 시 model="ft:..."
```

<br>

## 5. 추가 팁

### a) JSONL format 매우 중요

- "messages" 배열 구조만 허용
- assistant role과 user role 모두 있어야 함
- 불필요한 문장·주석 금지



### b) 데이터 품질이 성능의 대부분

- fine-tuning은 hyperparameter를 조절할 수 없음
- 따라서, "**데이터가 곧 모델의 능력**"

<br>

### c) 데이터 수는 적어도 50~100개는 추천

- 응답 스타일 위주라면 수십 개로도 효과 있음
- 복잡한 reasoning fine-tuning이라면 수천 개 사용

