---
title: LLM Libraries 정리
categories: [DLF, LLM, Python, MULT]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# LLM Libraries 정리

1. Hugging face의 라이브러리

- `transformers`

- `peft`

- `accelerate`

- `datasets`

2. `trl`

<br>

# 1. Hugging Face 관련 라이브러리

Hugging Face는 자연어 처리(NLP) 및 다양한 딥러닝 모델을 쉽게 활용할 수 있도록 지원하는 라이브러리를 제공한다.

<br>

## **(1) `transformers`**

- 사전 학습된 트랜스포머 모델(예: BERT, GPT, T5 등)을 쉽게 로드하고 파인튜닝할 수 있는 라이브러리.
- 토크나이저, 모델, 트레이너 기능을 포함.

<br>

### **예제 1: 사전 학습된 GPT-2 모델 로드 및 텍스트 생성**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "Deep learning is"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

<br>

### **예제 2: 사전 학습된 BERT를 활용한 문장 분류**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

text = "This is a great movie!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

logits = outputs.logits
predicted_class = torch.argmax(logits).item()
print(f"Predicted class: {predicted_class}")
```

<br>

## **(2) `peft` (Parameter-Efficient Fine-Tuning)**

- 대형 모델의 일부 가중치만 학습하는 방식(LoRA, Prefix-Tuning 등)을 지원하는 라이브러리.
- VRAM 사용량을 줄이고 효율적으로 파인튜닝 가능.

<br>

### **예제 1: LoRA를 활용한 BERT 파인튜닝**

```python
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1, task_type="SEQ_CLS"
)
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
```

<br>

### **예제 2: GPT-2에서 LoRA를 적용한 텍스트 생성**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1, task_type="CAUSAL_LM"
)
peft_model = get_peft_model(model, peft_config)

input_text = "Artificial intelligence is"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
output = peft_model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

<br>

## **(3) `accelerate`**

- 멀티-GPU, TPU, CPU 등에서 쉽게 모델을 학습할 수 있도록 도와주는 라이브러리.
- `DataParallel` 같은 PyTorch 기본 API보다 더 쉽게 사용할 수 있음.

<br>

### **예제 1: `accelerate`를 활용한 모델 훈련**

```python
from transformers import AutoModelForSequenceClassification
from accelerate import Accelerator

accelerator = Accelerator()
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

device = accelerator.device
model.to(device)
```

<br>

### **예제 2: 멀티-GPU에서 데이터 병렬 처리**

```python
import torch
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device

# 모델과 데이터를 accelerator에 할당
model = torch.nn.Linear(10, 2).to(device)
optimizer = torch.optim.Adam(model.parameters())
inputs = torch.randn(16, 10).to(device)

# 데이터 병렬 연산 적용
model, optimizer, inputs = accelerator.prepare(model, optimizer, inputs)
outputs = model(inputs)
print(outputs.shape)  # (16, 2)
```

<br>

## **(4) `datasets`**

- 다양한 공개 데이터셋을 쉽게 로드하고 처리할 수 있는 라이브러리.
- `pandas`, `torch`, `numpy` 등 다양한 형식으로 변환 가능.

<br>

**예제 1: `datasets`을 활용한 IMDb 데이터셋 로드**

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
print(dataset["train"][0])  # 첫 번째 리뷰 출력
```

<br>

**예제 2: `datasets`을 활용한 데이터셋 전처리 및 PyTorch 텐서 변환**

```python
from datasets import load_dataset
import torch

dataset = load_dataset("imdb")

# 텍스트를 토큰 길이에 따라 필터링
def filter_long_examples(example):
    return len(example["text"].split()) < 200

dataset = dataset.filter(filter_long_examples)

# PyTorch 텐서로 변환
tensor_dataset = dataset.with_format("torch")
print(tensor_dataset["train"][0])
```

<br>

# 2. `trl` (Transformers Reinforcement Learning)

- Hugging Face에서 강화 학습을 활용한 트랜스포머 모델 튜닝을 지원하는 라이브러리.
- 특히 **RLHF (Reinforcement Learning from Human Feedback)** 기반의 모델 최적화에 유용.

<br>

### **예제 1: GPT-2 모델을 PPO(PPOTrainer)로 훈련하는 기본 코드**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

config = PPOConfig(batch_size=8)
ppo_trainer = PPOTrainer(config, model, None, tokenizer)

queries = ["Hello, how are you?"] * 8
query_tensors = [tokenizer(q, return_tensors="pt").input_ids for q in queries]
responses = [model.generate(q, max_length=20) for q in query_tensors]

rewards = [1.0] * 8  # 임의의 보상 값
ppo_trainer.step(query_tensors, responses, rewards)
```

<br>

### **예제 2: SFT (Supervised Fine-Tuning) 활용한 미세 조정**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

dataset = load_dataset("tiny_shakespeare")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(output_dir="./results", per_device_train_batch_size=4, num_train_epochs=1)
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_datasets["train"])
trainer.train()
```

