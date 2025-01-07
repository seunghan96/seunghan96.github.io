# 1. LLM

Google Brain (2017): Transformer

Copilot (OpenAI + Microsoft)



## (1) LLM이란

Transformer arch

책의 목표:

- model selection
- data format
- fine-tuning params

에 대해 좋은 선택을 내릴 수 있도록!

<br>

### 1) LLM 정의

NLP의 하위 분야: 언어 모델링 (Language Modeling, LM)

2 종류의 LM

- (1) Autoregressive LM (e.g., GPT)
- (2) Autoencoding LM (e.g., BERT)

( 1+2: T5 )



|      LLM      | 디스크 크키 (~GB) | 메모리 사용량(~GB) | 파라미터<br />(~ 박만) | 훈련 데이터 |
| :-----------: | :---------------: | :----------------: | :--------------------: | :---------: |
|  BERT-Large   |        1.3        |        3.3         |          340           |     20      |
|  GPT-2 117M   |        0.5        |        1.5         |          117           |     40      |
|  GPT-2 1.5B   |         6         |         16         |          1500          |     40      |
|  GPT-3 175B   |        700        |        2000        |        175,000         |     570     |
|    T5-11B     |        45         |         40         |         11,000         |     750     |
| RoBERTa-Large |        1.5        |        3.5         |          355           |     160     |
| ELECTRA-Large |        1.3        |        3.3         |          335           |     20      |

<br>

### 2) LLM 주요 특징

Seq2Seq: Encoder + Decoder

- Encoder: 텍스트 이해
- Decoder: 텍스트 생성

<br>

### 3) LLM 작동 원리

Pretraining 

- BERT
  - Task: MLM + NSP
  - Dataset: Wikipedia + BookCorpus

<br>

Procedures

- Step 1) Pretraining
- Step 2) Fine-tuning (SL)

<br>

### 4) Others

- Attention
- Embedding
- Tokenization

<br>

### 5) Beyond LLM

Alignment + RLHF

- Alignment: 모델이 사용자의 기대에 부합하는 입력 prompt에 얼마나 잘 답변할 수 있는지
- RLHF: 인간 피대븍 기반 강화 학습

<br>

## (2) 많이 사용되는 LLM

- Google: BERT

  ( = Bidirectional Encoder Representation from Transformers )

  - Autoencoding LLM
  - Encoder-only arch

- OpenAI: GPT

  ( = Generative Pre-trained Transformer )

  - Autoregressive LLM
  - Decoder-only arch

- Google: T5

  ( = Text-to-Text Transfer Transformer )

  - w/o fine-tuning ($\leftrightarrow$ BERT. GPT2)

<br>

## (3) 도메인 특화 LLM

(e.g., BioGPT)

장점: 특정 text 집합에서의 훈련. (좁지만, 특화된 사전 훈련)

<br>

## (4) LLM을 이용한 애플리케이션

해당 책에서는 3가지 방식으로 LLM을 사용할 것.

- Pretrained LLM을 fine-tuning 없이 활용
  - e.g., Pretrain BERT/GPT를 통해 정보 검색 시스템 만들기
- Pretrained LLM을 fine-tuning하여 특정 task 특화하여 수행
  - e.g., T5를 fine-tuning하여 특정 도메인의 문서를 요약
- Pretrained LLM를 통해 합리적인 직관으로 수행할 수 있는 작업 요청
  - e.g., GPT4에 블로그 포스트 작성 요청

<br>

### 1) 전통적인 NLP 작업

- Text classification
- Translation
- SQL 생성

<br>

### 2) 자유로운 텍스트 생성

<br>

### 3) 정보 검색 / 신경망 의미 기반 검색

LLM을 매번 새로운 상태로 fine-tuning..?

정보를 동적으로 유지하기 위해, DB를 사용하여 "정보 검색"

<br>

### 4) Chatbot

<br>