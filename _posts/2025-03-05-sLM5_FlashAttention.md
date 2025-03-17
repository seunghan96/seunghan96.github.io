---
title: (sLM-5) Flash Attention
categories: [LLM, MULT, NLP]
tags: []
excerpt: Flash Attention 개념, 코드 실습
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
# Flash Attention

## Contents

1. Self-attention의 문제점
2. Flash Attention의 핵심
3. Flash Attention 코드
   1. Task 소개
   2. Load Model
   3. Inference

<br>

# 1. Self-attention의 문제점

Self-Attention을 계산할 때,

1. Query, Key, Value 행렬을 만들고
2. Query와 Key를 곱해서 Attention Score를 구한 후
3. Softmax를 적용하고 Value와 곱해 최종 출력을 얻음

이 과정에서 **Attention Score 행렬 (크기: `sequence_length × sequence_length`)을 메모리에 저장**해야!

$\rightarrow$ 시퀀스 길이가 길어질수록, 메모리 사용량이 기하급수적으로 증가하여 **연산 속도가 느려**지고 **GPU 메모리 부족** 문제가 발생!

<br>

# 2. Flash Attention의 핵심

한 줄 요약: **Self-Attention 연산**을 **더 빠르고 효율적으로 수행**하는 기술

<br>

How?  **메모리 접근을 최소화**하고, **GPU의 연산 자원을 최대한 활용**하는 방식!

$\rightarrow$ **큰 행렬을 한 번에 메모리에 로드하지 않고, 작은 "블록 단위"로 처리**

<br>

세부 아이디어

- (1) **온-칩 메모리(SRAM) 활용**: GPU의 빠른 캐시 메모리를 적극 활용하여 DRAM 접근을 최소화함
- (2) **블록 단위 연산**: Attention 행렬을 나누어 블록 단위로 연산하고, Softmax도 부분적으로 계산한 뒤 합치는 방식 사용
- (3) **Fusion 기법 적용**: 여러 개의 연산을 하나로 합쳐 불필요한 데이터 이동을 줄임

<br>

장점

- **메모리 절약**: Attention Score를 저장하지 않아도 되므로 **메모리 사용량이 3배 이상 감소**
- **속도 향상**: 기존 Self-Attention보다 **약 2~4배 빠름**
- **더 긴 시퀀스 처리 가능**: 메모리 부족 문제 없이 더 긴 문장을 처리 가능

<br>

# 3. Flash Attention 코드

설치하기

```bash
!pip install flash-attn==2.6.3
!pip install accelerate==0.30.1
!pip install transformers==4.39.3
```

<br>

## (1) Task 소개

**FlashAttention2를 Phi-2모델에 적용하기**

<br>

## (2) Load Model

```python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
```

```python
# 사용할 모델: Phi-2
model_id = "microsoft/phi-2"
```

<br>

`AutoModelForCausalLM.from_pretrained`의 인자로,

- `attn_implementation="flash_attention_2"`를 설정해주면 된다!

```python
# (1) Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True  
)

# (2) Model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",                        
    device_map="auto",                        
    attn_implementation="flash_attention_2",
    trust_remote_code=True
)
```

<br>

## (3) Inference

**a) Prompt 내용**

```python
prompt = '''def factorial(n):
    """
    Calculate the factorial of a number n
    """
'''
```

<br>

**b) Prompt를 tokenizing하기**

```python
input_ids = tokenizer(
    prompt,
    return_tensors="pt",         
).to(model.device)           
```

<br>

**c) Terminator**: 단어 생성 종결 조건 지정!

```python
terminators = [
    tokenizer.eos_token_id,
]
```

<br>

**d) 생성하기**

- `input_ids`: dictionary 형태이다!
- `do_sample`
  - True: 매번 다른 결과 (sampling)
  - False: Greedy decoding

```python
outputs = model.generate(
    **input_ids,  
    max_new_tokens=200,   
    eos_token_id=terminators,
    do_sample=False,    
    pad_token_id=tokenizer.eos_token_id  
)
```

<br>

**e) 결과 확인**

```python
response = outputs[0][input_ids['input_ids'].shape[-1]:]
print("response : ", tokenizer.decode(response, skip_special_tokens=True))
```

```
response :      if n == 0:
        return 1
    else:
        return n * factorial(n-1)

# Test the function
print(factorial(5)) # Output: 120
```

<br>

### Reference

- [패스트캠퍼스] 8개의 sLM모델로 끝내는 sLM 파인튜닝
