---
title: LLM 모델 파인튜닝을 위한 GPU 최적화 (4) 실습
categories: [DLF, LLM, PYTHON, MULT]
tags: []
excerpt: Single GPU 환경에서 LLM 돌리기
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# 4. Single GPU 실습

## 패키지 설치

```cmd
pip install -q -U bitsandbytes # Q-LoRA library
pip install datasets -U
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
pip install pandas
pip install wandb
```

<br>

## Contents

1. **Dataset**
   1. 패키지 불러오기
   2. Hugging face 로그인
   3. 데이터셋 불러오기
2. **Tokenizer**
   1. 패키지 불러오기
   2. tokenizer 불러오기
   3. tokenizer 전처리
   4. 그 외의 사항
3. **Prompt**
   1. Template
   2. Prompt
   3. Tokenize
   4. Prompt를 Tokenize
   5. Train & Val dataset
4. **Model**
   1. Model 소개
   2. Quantization Configuration
   3. Load Model
5. **LoRA**
   1. LoRA Configuration
   2. LoRA 적용하기
6. **LoRA 학습**
   1. Setting
   2. Train
   3. Trainer
   4. Save
   5. Hugging Face에 올리기

<br>

# 1. Dataset

## (1) **패키지 불러오기**

```python
from datasets import load_dataset
import pandas as pd
from huggingface_hub import login
```

<br>

## (2) **Hugging face 로그인**

```python

```

<br>

## (3) **데이터셋 불러오기**

```python
data_path = "DopeorNope/Ko-Optimize_Dataset"
data = load_dataset(data_path)
df = pd.DataFrame(data['train'])
```

```python
print(df.shape)
data
```

```
(10000, 3)
DatasetDict({
    train: Dataset({
        features: ['input', 'instruction', 'output'],
        num_rows: 10000
    })
})
```

<br>

경우에 따라서, input 없는 경우도 있다.

- input 예시: *귀하는 사람들이 정보를 찾도록 도와주는 AI 어시스턴트입니다. 사용자가 질문을 합니다*

```python
print(df.columns)
```

```
['input','instruction','output']
```

<br>

# 2. Tokenizer

## (1) **패키지 불러오기**

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import bitsandbytes as bnb
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training)

```

<br>

## (2) **(pretrained) tokenizer 불러오기**

```python
model_path = "beomi/Llama-3-Open-Ko-8B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
```

<br>

## (3) **tokenizer 전처리**

참고: ***LLama tokenizer: PAD token 없음***

- (1) LLaMA 모델의 기본 tokenizer는 `pad_token`을 따로 정의하지 않는다!

  (일반적으로 `pad_token`은 배치 단위 입력을 맞추기 위해 필요하지만, LLaMA는 원래 패딩 없이 동작하도록 설계됨!)

- (2) 따라서, `Trainer`를 사용할 때는 **EOS(Token End-of-Sequence)를 PAD로 활용**한다!
  - PAD 토큰이 없으므로 `pad_token_id`를 `eos_token_id`로 설정하여 패딩 효과를 내도록!
  - 이렇게 하면 **배치 내 길이가 다른 샘플을 맞출 때 EOS 토큰이 패딩 역할을 하게 됨**.
- (3) 하지만 `trl` 라이브러리 사용 시, 조금 다르다!
  - `trl` 라이브러리의 `SFTTrainer`는 `pad_token`을 EOS로 대체하지 않고,
    **"<|reserved_special_token_0|>"**이라는 새로운 특수 토큰을 `pad_token`으로 설정함.
  - 즉, `SFTTrainer`는 패딩을 위한 별도의 특수 토큰을 생성하고 이를 사용하도록 한다.

```python
# 토크나이저 세팅: QLoRA시 pad 토큰을 eos로 설정해주기
bos = tokenizer.bos_token_id
eos = tokenizer.eos_token_id
pad = tokenizer.pad_token_id
```

```python
tokenizer.pad_token_id = eos
# tokenizer.add_special_tokenizer.add_special_tokens({"pad_token":"<|reserved_special_token_0|>"}) # trl의 SFTTrainer 
tokenizer.padding_side = "right" # Mistral: Left
```

<br>

## (4) 그 외의 사항

- `train_on_inputs`: 
  - True: loss(input+output, input_pred+output_pred)
  - False: loss(output, output_pred)

```python
cut_off_len = 4098 # max context length
val_size = 0.005 # 보다 적절한 것은, validation set을 따로 구축하기.
train_on_inputs = False 
add_eos_token = False
```

<br>

# 3. Prompt

## (1) Template

답변 띄어쓰기 유의하기! (특히 `train_on_inputs=False` 인 경우)

- $$\because$$ X,Y부분을 나누는 단어 길이 셀 때 실수할 수 있음!

```python
# 답변에 띄어쓰기 X
template = {
    "prompt_input": "아래는 문제를 설명하는 지시사항과, 구체적인 답변을 방식을 요구하는 입력이 함께 있는 문장입니다. 이 요청에 대해 적절하게 답변해주세요.\n###입력:{input}\n###지시사항:{instruction}\n###답변:",
    "prompt_no_input": "아래는 문제를 설명하는 지시사항입니다. 이 요청에 대해 적절하게 답변해주세요.\n###지시사항:{instruction}\n###답변:"
}
```

<br>

## (2) Prompt

```python
from typing import Union

def generate_prompt(
    instruction: str,
    input: Union[None, str] = None,
    label: Union[None, str] = None,
    verbose: bool = False
) -> str:
    """
    주어진 instruction, input, label을 사용하여 프롬프트를 생성하는 함수.

    Parameters:
    - instruction (str): 문제 설명 또는 지시사항.
    - template (dict): 입력이 있는 경우와 없는 경우의 템플릿을 포함한 딕셔너리.
    - input (str or None): 문제에 대한 구체적인 입력 (옵션).
    - label (str or None): 정답 또는 응답 (옵션).
    - verbose (bool): 생성된 프롬프트를 출력할지 여부.

    Returns:
    - str: 완성된 프롬프트.
    """
    if input:
        res = template["prompt_input"].format(instruction=instruction, input=input)
    else:
        res = template["prompt_no_input"].format(instruction=instruction)

    if label:
        res = f"{res}{label}"

    if verbose:
        print(res)

    return res
```

<br>

## (3) Tokenize

역할: prompt가 들어오면, 이를 tokenizer를 사용하여 tokenize한다.

```python
def tokenize(prompt, add_eos_token=True):
   result = tokenizer(prompt,truncation=True,max_length=cut_off_len,padding=False,return_tensors=None,)
   if (result["input_ids"][-1] != tokenizer.eos_token_id
       and len(result["input_ids"]) < cut_off_len
       and add_eos_token
      ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

   result["labels"] = result["input_ids"].copy()
   return result
```

<br>

## (4) Prompt를 Tokenize

```python
def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"]
        )
    tokenized_full_prompt = tokenize(full_prompt)
    
    if not train_on_inputs:
        user_prompt = generate_prompt(data_point["instruction"], data_point["input"])
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

    return tokenized_full_prompt
```

<br>

## (5) Train & Val dataset

`generate_and_tokenize_prompt` 함수를 사용하여 train & validation data 만들기

```python
if val_size > 0:
  train_val = data["train"].train_test_split(test_size=val_size, shuffle=True, seed=42)
  train_data = (train_val["train"].shuffle().map(generate_and_tokenize_prompt))
  val_data = (train_val["test"].shuffle().map(generate_and_tokenize_prompt))
else:
  train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
  val_data = None
```

<br>

train & val 데이터를 확인해보자!

```python
train_data
```

```
Dataset({
    features: ['input', 'instruction', 'output', 'input_ids', 'attention_mask', 'labels'],
    num_rows: 9950
})
```

<br>

```python
val_data
```

```
Dataset({
    features: ['input', 'instruction', 'output', 'input_ids', 'attention_mask', 'labels'],
    num_rows: 50
})
```

<br>

# 4. Model

## (1) Model 소개

**Meta의 Llama 3.1 8B Instruct 모델**

- 80억 개의 parameter (경량화)
- 다국어 LLM
- 대화형 응용 프로그램을 위해 최적화
- pretrain + instruction-tuning

<br>

## (2) Quantization Configuration

**`BitsAndBytesConfig`를 사용해 4비트 양자화(quantization) 설정을 정의**

-  Hugging Face의 `BitsAndBytes`(bnb) 라이브러리를 활용하여 **메모리 효율적인 모델 로딩**을 수행

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4비트 양자화 사용
    bnb_4bit_use_double_quant=True,  # 더블 양자화 사용
    bnb_4bit_quant_type="nf4",  # 양자화 방식: NF4 사용
    bnb_4bit_compute_dtype=torch.bfloat16,  # 연산 시 데이터 타입: bfloat16
    bnb_4bit_quant_storage=torch.bfloat16,  # 저장 시 데이터 타입: bfloat16
)
```

- `load_in_4bit=True`

  - 모델을 **4비트 양자화된 형태로 로드**함.

  - 모델 가중치를 4비트로 변환하여 **메모리 사용량을 줄이고** 연산 속도를 향상시킴.

- `bnb_4bit_use_double_quant`

  - **Double Quantization(더블 양자화)** 적용.

  - 4비트 양자화를 한 번 더 양자화하여 **추가적인 메모리 절약 가능**.

- `bnb_4bit_quant_type="nf4"` 

  - `NF4`(Normalized Float 4) 방식 사용.
    - NF4는 4비트 양자화 방식 중 하나
    - 기존의 정규화되지 않은 4비트보다 **더 높은 표현력을 제공** 

  - LLaMA와 같은 모델에서 **양자화로 인한 성능 저하를 최소화**하는 데 유리

- `bnb_4bit_compute_dtype=torch.bfloat16`

  - 모델이 4비트로 저장되더라도 **연산은 `bfloat16`을 사용하여 수행**.

  - `bfloat16`(Brain Floating Point 16)은 `float16`과 비슷하지만 **더 넓은 표현 범위를 가짐**.

  - **LLM 훈련과 추론에서 널리 사용되는 데이터 타입**으로, 안정적이고 빠름.

- `bnb_4bit_quant_storage=torch.bfloat16` 

  - 양자화된 값이 **bfloat16 타입으로 저장됨**.

  - 저장은 `bfloat16`, 실제 연산은 `bfloat16`을 사용하여 **성능과 메모리 효율성 간 균형을 유지**.

<br>

## (3) Load Model

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config = quantization_config,
    torch_dtype = torch.bfloat16,
    device_map = {"" : 0}
    )
```

<br>

SFT를 할 경우

```python
# ( pad = tokenizer.pad_token_id  )
# model.config.pad_token_id = tokenizer.pad_token_id 
```

<br>

```python
model = prepare_model_for_kbit_training(model)
```

- quantized 모델 훈련을 위한 준비 과정
- 주로 **4비트 또는 8비트 양자화된 모델을 효율적으로 학습할 수 있도록 변경하는 역할**

<br>

# 5. LoRA

## (1) LoRA Configuration

```python
config = LoraConfig(
    r = 16,
    lora_alpha = 16,
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM"
    )
```

<br>

### LoRA 적용 대상 찾기 (사용 X, 단지 확인 O)

```python
def find_all_linear_names(model):
  cls = bnb.nn.Linear4bit  # 4비트 양자화된 선형 계층 클래스 지정
  lora_module_names = set() 
  for name, module in model.named_modules():  # 모든 서브 모듈을 순회
    if isinstance(module, cls):  # 만약 해당 모듈이 4비트 Linear 계층이면
      names = name.split('.')  # 계층 이름을 '.' 기준으로 분리
      lora_module_names.add(names[0] if len(names) == 1 else names[-1])  
  return list(lora_module_names)  
```

- 모델 내부에서 **`bnb.nn.Linear4bit` 계층을 찾아 해당 레이어 이름을 리스트로 반환**
- 주로 **LoRA 적용 시 필요한 레이어를 식별하는 데 사용**

<br>

```python
print('Trainable targer module:',find_all_linear_names(model))
```

```
Trainable targer module: ['up_proj', 'k_proj', 'q_proj', 'down_proj', 'v_proj', 'o_proj', 'gate_proj']
```

<br>

## (2) LoRA 적용하기

```python
model = get_peft_model(model, config)
```

<br>

`model`을 찍어보면, `lora_embedding_A`, `lora_embedding_B`가 끼여있음을 알 수 있음

```
model
```

```
생략
```

<br>

### 학습 가능 파라미터 확인

```python
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
```

```python
print_trainable_parameters(model)
```

```
trainable params: 13631488 || all params: 2809401344 || trainable%: 0.4852097059436731
```

<br>

# 6. LoRA 학습

## (1) Setting

- Mini-batch 크기를 1로 설정
- Gradient Accumulation을 8번 => 배치 크기 8

```python
num_epochs = 1
micro_batch_size = 1
gradient_accumulation_steps = 8

warmup_steps = 100
learning_rate = 5e-8
```

<br>

그 외의 사항들

```python
# 여러 텍스트 묶어서 사용
group_by_length = False

optimizer = 'paged_adamw_8bit'
beta1 = 0.9
beta2 = 0.95

lr_scheduler = 'cosine'
logging_steps = 1

use_wandb = True
wandb_run_name = 'Single_GPU_Optim'

use_fp16 = False
use_bf_16 = True
evaluation_strategy = 'steps'

eval_steps = 50
save_steps = eval_steps
save_strategy = 'steps'
```

<br>

```python
model.gradient_checkpointing_enable()
```

- **Gradient Checkpointing** 활성화
- **메모리 사용량을 줄이기 위해** 역전파 시 중간 활성화 값을 저장하지 않고, 필요할 때 다시 계산!

<br>

## (2) Trainer

```python
trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=TrainingArguments(
    per_device_train_batch_size = micro_batch_size,
    per_device_eval_batch_size = micro_batch_size,
    gradient_accumulation_steps = gradient_accumulation_steps,
    warmup_steps = warmup_steps,
    num_train_epochs = num_epochs,
    learning_rate = learning_rate,
    adam_beta1 = beta1, # adam 활용할때 사용
    adam_beta2 = beta2, # adam 활용할때 사용
    fp16 = use_fp16,
    bf16 = use_bf_16,
    logging_steps = logging_steps,
    optim = optimizer,
    evaluation_strategy = evaluation_strategy if val_size > 0 else "no",
    save_strategy="steps",  #스텝기준으로 save
    eval_steps = eval_steps if val_size > 0 else None,
    save_steps = save_steps,
    lr_scheduler_type=lr_scheduler,
    output_dir = output_dir,
    #save_total_limit = 4,
    load_best_model_at_end = True if val_size > 0 else False ,
    group_by_length=group_by_length,
    report_to="wandb" if use_wandb else None,
    run_name=wandb_run_name if use_wandb else None,
    ),
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
```

<br>

## (3) Train

```python
# eval시 True 추천
# 학습 시 (cache 활용 필요 없으므로) False 추천
model.config.use_cache = False

trainer.train()
```

<br>

## (4) Save

```python
output_dir='./llama_singleGPU-v1'

trainer.save_model()
tokenizer.save_pretrained(output_dir)
```

<br>

## (5) Hugging Face에 올리기

참고: 훈련한 것은 "전체 모델"이 아닌 "LoRA"이다.

$$\rightarrow$$ 따라서, 병합할 필요가 있다 "전체 모델 + LoRA"

```python
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(model_path, token=my_hf_key)
merged_model = PeftModel.from_pretrained(base_model, output_dir)
merged_model = merged_model.merge_and_unload()
merged_model.push_to_hub('seunghan96/Single_GPU_Llama3-8B')
tokenizer.push_to_hub('seunghan96/Single_GPU_Llama3-8B')
```

