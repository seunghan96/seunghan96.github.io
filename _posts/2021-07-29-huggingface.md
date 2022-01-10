---
title: Hugging Face \& Bert
categories: [NLP,ABSA]
tags: [NLP, ABSA]
excerpt: 
---

(참고 : Ready-To-Use Tech 유튜브 강의)

# 1. BERT의 4가지 형태의 fine-tuning

- 1) Sequence Pair Classification (문장 2개의 관계 파악)
- 2) Single Sentence Classification (문장 분류)
- 3) Question Answering (질의 응답)
- 4) Single Sentence Tagging (개체 분석)

<br>

# 2. Hugging Face

주로 Transformer 기반의 모델들

<br>

## 1) 좌측 메뉴 소개

- **(a) Summary of Tasks** : Fine-tuning 방법들 소개
  - 위에 참고

- **(b) Summary of the Models** : Pre-training 모델들 구조 소개
  - 1) Autoregressive models
  - 2) Autoencoding models
  - 3) Seq2Seq models
  - 4) Multimodal models
  - 5) Retrieval based models

- **(c) Model sharing and uploading** : 여러 사람들이 서로 모델 공유/공개
- **(d) Models**
  - BERT,ALBERT,BART,GPT,T5....
    - ex) BERT :
      - BertTokenizer
      - BertTokenizerFast
      - ...
      - BertForSequenceClassification ( Pytorch용 )
      - BertForNextSentencePrediction
      - ...
      - TFBertForSequenceClassificaiton ( TF용 )

<br>

## 2) Official한 Pre-trained models

Advanced Guides : Official한 모델

- **Pretrained Models**

  - Architecture 소개

    - **bert-base-uncased**
    - **bert-large-uncased**
    - ...
    - **bert-base-multilingual-uncased**
    - **bert-base-multilingual-cased**

    - ...
    - **gpt-large**
    - ...

  - cased : 대소문자 구분 O

  - uncased : 대소문자 구분 X

  - base & large : 모델 size

<br>

여기서 pre-trained 모델을 받아와서, 나의 data/task에 맞게 fine-tuning하자!

<br>

## 3) Pre-train & Fine-tune 과정 규격화

example )

BART

```python
from transformers import BartModel
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer

# 1) tokenizer
kobart_tokenizer = get_kobart_tokenizer()

# 2) pre-trained model
model = BartModel.from_pretrained(get_pytorch_kobart_model())
```

<br>

Electra

```python
from transformers import ElectraTokenizerFast, ElectraModel,TfElectraModel

# 1) tokenizer
electra_tokenizer = ElectraTokenizerFast.from_pretrained('kykim/electra-kor-base')

# 2) pre-trained model
## Pytorch
model_pt = ElectraModel.from_pretrained('kykim/electra-kor-base')
## TF
model_tf = TFElectraModel.from_pretrained('kykim/electra-kor-base')
```

<br>

## 4) Example : GPT3로 문장 생성

- 1) library 불러오기 ( pre-trained 모델 & tokenizer )

```python
from transformers import BertTokenizerFast,TFGPT2LMHeadModel,GPT2LMHeadModel
```

<br>

- 2) tokenizer & model 불러오기 

```python
tokenizer = BertTokenizerFast.from_pretrained('kykim/gpt3-kor-small-based-on-gpt2')
model = GPT2HeadModel.from_pretrained('kykim/gpt3-kor-small-based-on-gpt2',pad_token_id=0)
```

<br>

- 3) input & output
  - xxx_text : 텍스트
  - xxx_ids : 해당 텍스트에 해당하는 token 들

```python
input_text = '인생이'
input_ids = tokenizer.encode(input_text,return_tensors='pt')
input_ids = input_ids[:,1:] # (0번쨰) cls 제거

output_ids = model.generate(input_ids)
output_text = tokenizer.decode(output_ids[0],skip_tokens=True)

print(output_text)
```

```python
'인생이 너무 행복하고 행복하다'
```

