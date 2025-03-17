---
title: (sLM-4) Quantization 실습
categories: [LLM, MULT, NLP]
tags: []
excerpt: 

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Quantization 실습

## Contents 

1. Quantization 시작하기
2. Quantization 코드
   1. Load Model
   2. Inference (Single)
   3. Inference (Batch)

<br>

# 1. Quantization 시작하기

**최근 트렌드**

- (X) 작은 모델을 그대로 사용하기
- (O) 큰 모델을 양자화하여 사용하기

<br>

패키지 설치하기

- `bitsandbytes`: 양자화를 위해 사용되는 패키지

```bash
!pip install bitsandbytes==0.43.1
!pip install accelerate==0.30.1
!pip install transformers==4.39.3
```

<br>

# 2. Quantization 코드

## (1) Load Model

```python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
```

```python
# 사용할 모델: LLaMA-3-8B-Instruct
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
```

<br>

**양자화를 위한 configuration**

```python
config = BitsAndBytesConfig(
    load_in_4bit=True,                    
    bnb_4bit_quant_type="nf4",            
    bnb_4bit_use_double_quant=True,       
    bnb_4bit_compute_dtype=torch.bfloat16 
)
```

<br>

`AutoModelForCausalLM.from_pretrained`의 인자로,

- `quantization_config=config`를 설정해주면 된다!

```python
# (1) Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# (2) Model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",        
    device_map="auto",         
    quantization_config=config  
)
```

<br>

## (2) Inference (Single)

**a) Prompt 내용**

```python
messages = [
    {"role": "system", "content": "You are a kind robot."},
    {"role": "user", "content": "이순신이 누구야?"},
]
```

<br>

**b) Prompt를 tokenizing하기**

- `add_generation_prompt`: 주어진 prompt의 뒤에, **새롭게 생성을 요구하는 token**을 넣을지!

```python
input_ids = tokenizer.apply_chat_template(
    messages2,
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True
).to(model.device)
```

<br>

**c) Terminator**: 단어 생성 종결 조건 지정!

```python
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
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
    input_ids=input_ids,      
    max_new_tokens=300,       
    eos_token_id=terminators, 
    do_sample=True,           
    temperature=0.7,          
    no_repeat_ngram_size=2,   
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
response :  ! (Sejong the Great was the fourth king of the Joseon Dynasty in Korea. He is known for his efforts to promote education and culture, and is considered one of Korea's most important monarchs.)
```

<br>

## (3) Inference (Batch)

**a) Prompt 내용**

```python
messages1 = [
    {"role": "system", "content": "You are a kind robot."},
    {"role": "user", "content": "이순신이 누구야?"},
]

messages2 = [
    {"role": "system", "content": "You are a kind robot."},
    {"role": "user", "content": "세종대왕이 누구야?"},
]
```

<br>

**b) Prompt를 tokenizing하기**

```python
prompt1 = tokenizer.apply_chat_template(
    messages1,
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=False
)

prompt2 = tokenizer.apply_chat_template(
    messages2,
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=False
)
```

<br>

**c) Terminator**

```python
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
```

<br>

**d) Tokenizer 세팅**

```python
tokenizer.padding_side = 'left'
tokenizer.pad_token_id = tokenizer.eos_token_id
```

<br>

**e) Prompt를 배치화 (여러 개 묶기)**

- 추가로, 여러 prompt를 묶은 이후의 "input_id"를 추출한다

```python
prompt_batch = [prompt1, prompt2]
input_ids_batch = tokenizer(prompt_batch, return_tensors='pt',padding="longest")['input_ids']
```

<br>

**f) 생성하기**

```python
outputs = model.generate(
    input_ids=input_ids_batch,
    max_new_tokens=30,        
    eos_token_id=terminators, 
    do_sample=True,           
    temperature=0.7,          
    no_repeat_ngram_size=2,   
    pad_token_id=tokenizer.eos_token_id  
)
```

<br>

**g) 결과 확인하기**

- for loop을 돌며, 여러 질문(prompt)에 대한 대답들을 각각 확인한다!

```python
for i, output in enumerate(outputs):
    response = output[input_ids_batch[i].shape[-1]:]
    print(f"response {i + 1}: ", tokenizer.decode(response, skip_special_tokens=True))
```

```
response 1:  I think you're asking who Yi Sun-sin is!

Yi Sun-shin (1545-1598) was a Korean admiral and
response 2:  I'm happy to help! 

Sejong the Great (1396-1450) was the fourth king of the Joseon Dynasty in
```

<br>

### Reference

- [패스트캠퍼스] 8개의 sLM모델로 끝내는 sLM 파인튜닝
