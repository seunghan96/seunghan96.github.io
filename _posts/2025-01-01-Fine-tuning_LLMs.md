---
title: Fine-tuning LLMs
categories: [LLM, MULT]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Fine-tuning LLMs

출처: https://wikidocs.net/253228

<br>

## Contents

1. Download
2. Preprocessing
3. Model Components
4. Fine-tuning LLMs
5. DPO (Direct Preference Optimization)
6. ORPO (Odds Ratio Preference Optimization)

<br>

# 1. Download

## 1-1. `datasets` 라이브러리

1. Download dataset 

```python
from datasets import load_dataset

repo_id = "jtatman/python-code-dataset-500k"

# total & train split
dataset_total = load_dataset(repo_id)
dataset_train = load_dataset(repo_id, split='train')
```

<br>

2. Dataset structure

- `DatasetDict` 객체 안에 여러 split 존재 (e.g., `train`)

```python
dataset
```

```
DatasetDict({
    train: Dataset({
        features: ['output', 'instruction', 'system'],
        num_rows: 559515
    })
})

```

<br>

3. Read file

- 아래 예시) SFT 양식: `output`, `instruction`, `system`

```python
dataset_train = load_dataset(repository_id, split='train')
dataset_train[0]
```

```
{'output': '...',
 'instruction': '...',
 'system': '...'}
```

<br>

## 1-2. `hugging_face` 라이브러리

1. Download dataset

```python
from huggingface_hub import hf_hub_download

repo_id = "jtatman/python-code-dataset-500k"
file_nm = "data/train-00000-of-00002.parquet"

# 반환값: 경로
file_path = hf_hub_download(repo_id=repo_id, 
                            filename=file_nm, 
                            repo_type="dataset")
```

<br>

2. Load file

```python
df = pd.read_parquet(file_path, engine='pyarrow')
```

<br>

# 2. Preprocessing

## 2-1. `map` function

```python
def create_conversation(sample):
  return {
    "messages": [
      {"role": "system", "content": sample["system"]},
      {"role": "user", "content": sample["instruction"]},
      {"role": "assistant", "content": sample["output"]}
      ]
    }

processed_dataset = dataset.map(create_conversation)
```

참고) 각 샘플의 형식:

```
{'output': '...',
 'instruction': '...',
 'system': '...'}
```

<br>

## 2-2. `filter` function

```python
def filter_function(sample):
  return len(sample['text']) <= 50 

filtered_dataset = dataset.filter(filter_function)
```

<br>

## 2-3. Add/Remove columns

(1) Remove: `remove_columns()`

```python
dataset = dataset.remove_columns('system')
```

<br>

(2) Add

```python
def add_length_column(example):
  example['length'] = len(example['text'])
  return example 

dataset = dataset.map(add_length_column)
```

<br>

## 2-4. Split & Merge

(1) Split

```python
train_test = dataset.train_test_split(test_size=0.2) 
test_valid = train_test['test'].train_test_split(test_size=0.5)

datasets = {
    'train': train_test['train'],
    'validation':test_valid['train'],
    'test': test_valid['test'] 
}
```

<br>

(2) Merge

```python
from datasets import concatenate_datasets

combined_dataset = concatenate_datasets([dataset1, dataset2])
```

<br>

## 2-5. Shuffle

```python
shuffled_dataset = dataset.shuffle(seed=777)
```

<br>

## 2-6. Batchify

```python
def tokenize_function(examples):
  return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, 
                                batched=True, 
                                batch_size=1000)
```

<br>

## 2-7. Tokenize

```python
from transformers import AutoTokenizer

hf_or_local_model = 'dazare/ggobugi-llama3-v4'
tokenizer = AutoTokenizer.from_pretrained(hf_or_local_model)

def create_conversation(sample):
  return {
    "messages": tokenizer.apply_chat_template([
      {"role": "system", "content": sample["system"]},
      {"role": "user", "content": sample["instruction"]},
      {"role": "assistant", "content": sample["output"]}
      ], tokenize=False, add_generation_prompt=False)
    }

processed_dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)
```

<br>

# 3. Model Components

## (1) Models

```python
from transformers import AutoModelForCausalLM

hf_or_local_model = 'dazare/ggobugi-llama3-v4'
model = AutoModelForCausalLM.from_pretrained(hf_or_local_model)
```

```python
layer_names = model.state_dict().keys()

for name in layer_names:
    print(name)
```

```
model.embed_tokens.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.self_attn.rotary_emb.inv_freq
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.input_layernorm.weight
model.layers.0.post_attention_layernorm.weight
...

model.norm.weight
lm_head.weight
```

- `{identifier}.{layer}.{layer_number}.{component}.{module}.{parameter}`

<br>

## (2) Optimizers

#### AdamW (8 bit)

- Parameter & Gradient: **8 bit**
- **8bit 저장**으로 인해 상당한 메모리 절감

<br>

#### PagedAdamW (32 bit)

- Parameter & Gradient: **32 bit**
- **메모리 페이징**을 통해 메모리를 더 효율적으로 관리

<br>

# 4. Fine-tuning LLMs

## (1) SFT (Supervised Fine Tuning)

Dataset: 아래의 세 가지로 구성

- (1) `system`
- (2) `user`
- (3) `assistant`

<br>

참고) HF에 올라와 있는 한국어 데이터셋

- `user`와 `assistant`만 포함된 경우도 종종 있음

<br>

Details

- a) QLoRA
- b) Model 호출 & PEFT 적용
- c) Tokenizer
- d) LoRA module
- e) Training arguments

<br>

### a) QLoRA

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16,
)
```

- `BitsAndBytesConfig` : (`transformers`의 클래스 중 하나로) Quantization을 위해 사용
- `load_in_4bit = True` : Linear layer의 **nf4/FP4 로 대체**하여 4bit quantiazation 설정
- `bnb_4bit_quant_type = "nf4"` : **Linear layer를 nf4로 대체**하여 4bit quantiazation 설정
- `bnb_4bit_compute_dtype = torch.bfloat16` : 연산 방법 = bf16

<br>

### b) Model 호출 & PEFT 적용

```python
model_id = "meta-llama/Llama-2-7b-chat-hf"   # HF hub 모델

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype = torch.bfloat16, 
    quantization_config = bnb_config, 
    device_map = "auto")
```

- ` device_map = "auto"`
  - Hugging Face의 **`accelerate`** 라이브러리에서 제공하는 기능
  - ***어떤 GPU/CPU***에 ***어느 layer***를 배치할지를 자동으로 결정
- `torch_dtype=torch.bfloat16`
  - 양자화되지 않은 부분은 그대로 torch_dtype 값으로 로드

<br>

```python
model.config.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)  
```

- `model.config.use_cache = False`: 

  - Transformer 모델에는 past_key_values 캐시를 저장해 두고 inference 시 속도 높임

  - 하지만 **학습 (특히 gradient checkpointing)** 시, 이 cache가 backprop에 방해가 되므로 꺼줘야!

- `model.gradient_checkpointing_enable()`:

  - gradient checkpointing 사용

    - 사용 X: 중간 activation을 메모리에 저장했다가 backward 때 그대로 사용
    - 사용 O: 중간 activation을 **일부만 저장**하고, backward 시에 필요한 부분을 **forward를 다시 실행해서 재계산**

  - 장/단점

    - 장점: GPU 메모리 사용량 ↓

    - 단점: 계산량은 ↑ (속도 조금 느려짐)

  - **큰 모델을 작은 GPU에서 학습할 때 필수적**

- `model = prepare_model_for_kbit_training(model)`

  - k-bit 양자화된 모델(예: 4bit, 8bit)을 **LoRA로 fine-tuning하기 위한 준비**

  - (i.e., 양자화된 Linear layer를 학습 가능하도록 **래핑**)

<br>

### c) Tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

`AutoTokenizer.from_pretrained(model_id)`

- HF에서 **model에 맞는 Tokenizer를 자동으로** 불러오는 클래스
- e.g., `model_id = "meta-llama/Llama-2-7b-hf"`
  - 이) 해당하는 tokenizer 설정 파일 (`tokenizer.json`, `special_tokens_map.json` 등)을 load

<br>

`tokenizer.pad_token = tokenizer.eos_token`

- 많은 **Causal LM (GPT/LLaMA 계열)** 모델은 pad_token을 따로 정의 X

- But 학습/추론 시 batchify 시, sequence 길이를 맞춰야 함 

  $$\rightarrow$$  **padding token**이 필요 (보통은 pad_token = eos_token으로 지정)

<br>

`tokenizer.padding_side = "right"`

- "right" → 시퀀스 **"끝"**에 pad 토큰 추가: [A,B,C,EOS,`PAD`,`PAD`]
- "left" → 시퀀스 **"앞"**에 pad 토큰 추가: [`PAD`,`PAD`,A,B,C,EOS]

- Causal LM에서는 **right padding**을 사용하는 것이 일반적

  ($$\because$$ attention mask와 causal mask가 **뒤쪽 padding을 무시하도록 짜여 있어서**)

<br>

### d) LoRA module

```python
peft_config = LoraConfig(
    r = 128,
    lora_alpha = 16,
    target_modules = find_all_linear_names(model),
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
print_trainable_parameters(model)
```

- `r` : LoRA의 rank
- `lora_alpha`: LoRA의 scaling factor
- `target_modules`: traininable modules
- `lora_dropout` : LoRA의 dropout
- `bias`: LoRA의 bias 여부 ('none', 'all', 'lora_only')

<br>

### e) Training arguments

```python
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    num_train_epochs=15,
    learning_rate=2e-4,
    bf16=True,
    save_total_limit=3,
    logging_steps=10,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_steps=8000,
    report_to='tensorboard'
)
```

<br>

### f) SFT

```python
trainer = SFTTrainer(
    model,
    train_dataset=datasets,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=4096,
    args=training_args,
)

trainer.train()
trainer.save_model(output_dir)
```

<br>

### g) Save

```python
output_dir = os.path.join(output_dir, "llama2_checkpoint")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
```

- **Eval_samples_per_second**: 초당 처리 샘플 수 → 클수록 추론 속도 ↑
- **Grad_norm**: 그라디언트 크기 → 0.01~10 범위가 안정적
- **Eval_steps_per_second**: 초당 평가 스텝 수 → 클수록 빠른 추론

<br>

# 5. DPO (Direct Preference Optimization)

## (1) 개요

***“RL 없이 RLHF 품질을”***

- **목표**: "사람이 더 선호한 응답"을 하도록 언어모델 (Policy)을 FT
- **데이터**: (prompt $$x$$, chosen $$y⁺$$, rejected $$y⁻$$) 형태의 **쌍(pair)** 
- **핵심 아이디어**: ***chosen prob > rejected prob를 "directly" 학습***
- **장점**: PPO처럼 환경/롤아웃/가치함수(critic)가 **없음** → 구현/안정성/속도 ↑.

<br>

## (2) 수식

- (학습) 정책: $$\pi_\theta(y|x)$$
- (참조) 정책: $$\pi_{\mathrm{ref}}(y|x)$$ (e.g., **Freezed SFT model**)
- **Logit 차이**를 **Preference**로 연결 (Bradley–Terry):
  - $$P_\theta(y^+ \succ y^- \mid x) \;=\; \sigma\!\left( \beta \big[(\log\pi_\theta(y^+|x) - \log\pi_\theta(y^-|x)) - (\log\pi_{\mathrm{ref}}(y^+|x) - \log\pi_{\mathrm{ref}}(y^-|x))\big] \right)$$.
- Loss function: $$\mathcal{L}_{\text{DPO}} \;=\; -\log \sigma\!\left( \beta \big[\Delta\log\pi_\theta - \Delta\log\pi_{\mathrm{ref}}\big] \right)$$.
  - $$\Delta\log\pi(\cdot)=\log\pi(y^+|x)-\log\pi(y^-|x)$$.

<br>

## (3) 구성 요소

1. **데이터**: $$(x, y⁺, y⁻)$$ 쌍. 
   - 수집 by Human/AI preference
2. **참조 정책** $$\pi_{\mathrm{ref}}$$: Freeze된 SFT model
3. **정책** $$\pi_\theta$$: Update할 현재 model (feat. LoRA)
4. **마스킹**: log prob 합산은 **응답 토큰($$y$$) 부분만** 사용 (i.e., prompt token/input 제외)

<br>

## (4) Code

```python
# 필요한 라이브러리 종류 
# "transformers[sentencepiece]==4.37.2" \
#  "datasets==2.16.1" \
#  "accelerate==0.26.1" \
#  "evaluate==0.4.1" \
#  "bitsandbytes==0.42.0" \
#  "trl==0.7.11" \
#  "peft==0.8.2" \
#  "pillow"

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets

import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments

from peft import LoraConfig

from trl import DPOTrainer
```

생략

<br>

# 6. ORPO (Odds Ratio Preference Optimization)

**odds ratio** (승산비)를 기반으로 선호 최적화

<br>

## (1) 개요

- **목적**: 사람 선호 데이터를 이용해 LM을 fine-tune
- **아이디어**: “좋은 답 $$y^+$$“과 “나쁜 답 $$y^-$$“의 **"odds ratio"** 가 사람이 원하는 방향으로 크도록 directly 최적화
- **데이터**: (DPO와 동일) **선호 데이터 (pairwise preference)** 

<br>

## (2) Odds Ratio

- 확률 p가 주어졌을 때, **odds(승산)** = $$\text{odds}(p) = \frac{p}{1-p}$$.
- 두 사건 A, B의 **odds ratio** = $$\frac{p_A/(1-p_A)}{p_B/(1-p_B)}$$.
- 해석: 사건 A가 사건 B보다 얼마나 더 “일어나기 쉬운가”를 odds 관점에서 보는 것

<br>

## (3) 수식

- 데이터: $$(x, y^+, y^-)$$ 
- 모델: $$\pi_\theta(y|x)$$
- ORPO의 목표: $$\max_\theta \; \log \frac{ \frac{\pi_\theta(y^+|x)}{1-\pi_\theta(y^+|x)} } { \frac{\pi_\theta(y^-|x)}{1-\pi_\theta(y^-|x)} }$$.
- 즉, **좋은 응답의 odds가 나쁜 응답 odds보다 커지도록** 하는 것.

<br>

## (4) DPO vs. ORPO

[DPO] SFT $$\rightarrow$$ DPO

[ORPO] ORPO 단독

- 뛰어난 퍼포먼스 & 상대적으로 적은 데이터와 메모리
- DPO처럼 보상모델 없이 학습하는데, 로짓 차이 대신 **odds ratio** 개념을 활용

<br>

| **항목**       | **DPO**                           | **ORPO**                                           |
| -------------- | --------------------------------- | -------------------------------------------------- |
| 핵심 수식      | 로그확률 차이(logit diff)         | odds ratio                                         |
| 참조 정책 필요 | ✅ (SFT reference)                 | ❌ (reference 불필요)                               |
| 직관           | “좋은 답의 로그확률 ↑, 나쁜 답 ↓” | “좋은 답 odds / 나쁜 답 odds ↑”                    |
| 장점           | 보상모델, PPO 불필요 / 안정적     | reference 모델 불필요 → 훨씬 간단                  |
| 단점           | 여전히 참조 모델 로딩 필요        | odds 기반이므로 일부 데이터셋에서 안정성 이슈 가능 |

즉 **ORPO는 DPO보다 더 간단**합니다. 참조 정책(ref model)을 안 쓰고 현재 모델 확률만으로 odds ratio를 직접 최적화합니다.

<br>

## (5) 참조모델 유무 (DPO: O vs. ORPO: X)

핵심 차이는 **로스 함수 설계 방식**과 **확률의 스케일(크기) 제어 방법**에서 옵니다.

### a) DPO: 필요 O

- DPO(Direct Preference Optimization)의 핵심 로스는:

  $$\mathcal{L}{DPO} = -\log\sigma\!\Big(\beta \big[(\log \pi\theta(y^+|x) - \log \pi_\theta(y^-|x)) - (\log \pi_{\text{ref}}(y^+|x) - \log \pi_{\text{ref}}(y^-|x))\big]\Big)$$

- 여기서 **참조 모델** $$\pi_{\text{ref}}$$ 은 보통 SFT 모델을 그대로 freeze 한 것입니다.

- 이유:

  1. **KL regularization 역할**: 현재 policy가 기존 SFT 모델에서 너무 멀리 벗어나지 않도록 기준(anchor)을 줍니다.
  2. **확률 스케일 안정화**: 언어모델의 로그확률은 토큰 길이나 분포에 따라 값 범위가 크게 변할 수 있는데, 참조 모델 대비 차이를 쓰면 그 스케일 문제를 완화합니다.
  3. **“선호 차이”에만 집중**: 절대적인 확률 값 대신 “정책 vs 참조”의 상대적 로그확률 차이를 최적화함으로써, 기존 모델의 지식을 유지하면서 선호 방향으로만 살짝 조정합니다.

즉, DPO는 **reference model을 “anchor”로 두어 KL 제약을 암묵적으로 구현**한 구조예요.

<br>

### b) ORPO: 필요 X

- ORPO(Odds Ratio Preference Optimization) 로스는:

  $$\mathcal{L}{ORPO} = -\log\sigma\!\Big(\beta \big[\log \tfrac{\pi\theta(y^+|x)}{1 - \pi_\theta(y^+|x)} - \log \tfrac{\pi_\theta(y^-|x)}{1 - \pi_\theta(y^-|x)}\big]\Big)$$.

- 여기서는 오직 **현재 모델** $$\pi_\theta$$ 의 확률만 사용합니다.

- 이유:

  1. **odds ratio 변환** 자체가 확률 스케일을 정규화하는 효과를 가집니다.
     - $$p \mapsto \frac{p}{1-p}$$는 “이 답을 선택할 odds”로 바꿔주기 때문에 절대값보다는 상대적 크기를 안정적으로 비교할 수 있음.
  2. 따라서 **별도의 anchor(참조 모델)** 없이도 “좋은 답 odds > 나쁜 답 odds”라는 학습 목표를 바로 세울 수 있습니다.
  3. KL 제약 대신 odds ratio 구조 자체가 과도한 분포 왜곡을 막는 regularizer 역할을 어느 정도 합니다.

<br>

| **방법** | **참조 모델 필요?** | **이유**                                                     |
| -------- | ------------------- | ------------------------------------------------------------ |
| **DPO**  | ✅ 필요              | (1) KL anchor 제공, (2) 확률 스케일 안정화, (3) 기존 SFT에서 크게 벗어나지 않도록 제어 |
| **ORPO** | ❌ 불필요            | (1) odds ratio 변환 자체가 정규화 역할, (2) anchor 없이도 직접 odds 비교로 선호 최적화 가능 |

- **DPO**는 KL 제약을 암묵적으로 넣기 위해 **참조 모델**이 필요하다.
- **ORPO**는 **odds ratio** 변환으로 안정성을 확보하기 때문에 **참조 모델이 필요 없다.**




