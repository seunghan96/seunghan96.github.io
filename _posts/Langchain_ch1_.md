# Langchain

## (0) Intro

### a) Langchain이란

- 개념: **LLM을 활용**해 다양한 애플리케이션을 개발할 수 있는 framework

- 두 가지 기능: 
  - (1) **문맥 인식**
  - (2) **추론**

- 활용 예시: RAG 어플리케이션 제작, 구조화된 데이터 분석, 챗봇 개발

<br>

### b) 설치

```bash
# v1) standard
pip install -r https://raw.githubusercontent.com/teddylee777/langchain-kr/main/requirements.txt

# v2) mini
pip install -r https://raw.githubusercontent.com/teddylee777/langchain-kr/main/requirements-mini.txt
```

<br>

### c) 구성

- **[LangChain 라이브러리](https://python.langchain.com/v0.2/docs/introduction/)**: 
  - **Python 및 JavaScript 라이브러리**
- **[LangChain 템플릿](https://templates.langchain.com/)**: 
  - 다양한 작업을 위한 쉽게 배포할 수 있는 **참조 아키텍처 모음**
- **[LangServe](https://github.com/langchain-ai/langserve)**: 
  - **REST API**로 배포하기 위한 라이브러리
- **[LangSmith](https://smith.langchain.com/)**: 
  - (LLM-agnostic) 구축된 chain 을 **디버그, 테스트, 평가, 모니터링**
- **[LangGraph](https://python.langchain.com/docs/langgraph)**: 
  - LLM을 사용한 상태유지가 가능한 다중 액터 애플리케이션을 구축하기 위한 라이브러리

<br>

### d) 개발 용이성

1. **컴포넌트의 조립 및 통합** 

- 모듈식으로 설계되어서 사용하기 편리

2. **즉시 사용 가능한 "체인"**

- 체인 = 컴포넌트의 내장 조합
- 개발 과정 간소화 가능

<br>

### e) 주요 모듈

1. **모델 I/O**

- 프롬프트 관리 및 최적화

2. **검색**

- RAG 통해 외부에서 데이터 가져오기

3. **에이전트**

- LLM이 어떠한 조치를 취할지 결정 & 실행 & 관찰 & 반복

<br>

## (1) 설치하기

https://wikidocs.net/257836

https://www.youtube.com/watch?v=mVu6Wj8Z7C0



## (2) OpenAPI 키 발급 및 테스트

https://wikidocs.net/233342



## (3) LangSmith 추적 설정

LangSmith: **LLM 애플리케이션 개발, 모니터링 및 테스트** 를 위한 플랫폼

<br>

### a) LangSmith의 추적 기능

**When useful?**

- 예상치 못한 결과 디버깅
- agent가 looping되는 경우
- chain이 예상보다 느린 경우
- agent가 각 단계에서 사용한 token 수 파악



**Details**

- (1) **"project" 단위**로 추적 가능
  - ( e.g., 실행 카운트, error 발생률, token 사용량 등 )
- (2) 프로젝트 click시, (해당 프로젝트 내에서 실행했던) **"모든 run 확인 가능"**
- (3) 검색 결과, **"LLM의 input/output"**등을 전부 상세히 기록

<br>

### b) LangSmith 추적 사용하기

- Step 1) **LangSmith API key 발급**
  - https://smith.langchain.com/ 회원가입
  - Setting - Personal - Create API Key
  - (주의: Key 보관 잘 하기)
- Step2) **`.env` 파일에 API Key & Project 정보 입력**
  - `LANGCHAIN_TRACING_V2`: **true** = 추적 허용
  - `LANGCHAIN_ENDPOINT`: `https://api.smith.langchain.com` : 그대로 두기
  - `LANGCHAIN_API_KEY`: **{API_KEY}**
  - `LANGCHAIN_PROJECT`: **프로젝트 명** 

<br>

### c) 추적 활성화하기

매우 간단! 환경 변수만 설정하면 됨.

```python
# 앞서 설정한 .env 불러오기
from dotenv import load_dotenv
load_dotenv()
```

<br>

( 만약, 프로젝트 명/추적 여부 변경 원하면, **직접 jupyter notebook에서도 변경 가능** )

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "LangChain 프로젝트명"
os.environ["LANGCHAIN_API_KEY"] = "LangChain API KEY 입력"
```

<br>

### c) Langchain의 편리한 사용

`langchain-teddynote` 패키지 ( 출처: https://wikidocs.net/250954 )

```bash
pip install langchain-teddynote
```

```python
from langchain_teddynote import logging

# Tracing On/Off
logging.langsmith("LangChain 프로젝트명") # On
logging.langsmith("LangChain 프로젝트명", set_enable=False) # Off
```

<br>

## (4) OpenAI의 API 사용 (GPT-4o 멀티모달)

### a) Environment setting

```python
from dotenv import load_dotenv
load_dotenv() 

from langchain_teddynote import logging
logging.langsmith("CH01-Basic")
```

<br>

### b) ChatOpenAI

**ChatOpenAI** = OpenAPI의 LLM

- `temperature`: sharpness 조절
  - Smaller = 더 집중 (다양성 low)
  - Higher = 덜 집중 (다양성 high)
- `max_tokens`: 최대 토큰 수
- `model_name`: 사용할 모델 지정
  - e.g., `gpt-3.5-turbo` - `gpt-4-turbo` - `gpt-4o`
  - 참조: https://platform.openai.com/docs/models

<br>

```python
from langchain_openai import ChatOpenAI

# LLM
llm = ChatOpenAI(
    temperature=0.1,  
    model_name="gpt-4o",
)

# Question
question = "대한민국의 수도는 어디인가요?"
response = llm.invoke(question)
```

<br>

### c) 답변의 형식 (AI Message)

```python
print(response) # All
print(response.content) # Answer
print(response.response_metadata) # Metadata
```

<br>

### d) LogProb 활성화

**Log probability of tokens**

( = LLM이 각 토큰을 예측할 확률 )

```python
llm_with_logprob = ChatOpenAI(
    temperature=0.1, 
    max_tokens=2048, 
    model_name="gpt-3.5-turbo", 
).bind(logprobs=True)
```

<br>

```python
question = "대한민국의 수도는 어디인가요?"

response = llm_with_logprob.invoke(question)
print(response.response_metadata) # Metadata (X), Log probability (O)
```

<br>

### e) 스트리밍 출력 (실시간)

```python
answer = llm.stream("대한민국의 아름다운 관광지 10곳과 주소를 알려주세요!")

# Option 1)
for token in answer:
    print(token.content, end="", flush=True)
```

<br>

(혹은, `langchain_teddynote` 패키지 활용할 경우)

```python
from langchain_teddynote.messages import stream_response

# Option 2)
stream_response(answer)
```

<br>

### f) Multimodal model

(e.g., text, image, audio, video ... )

Note) `gpt-4o`, `gpt-4-turbo`: 이미지 인식 기능 O

```python
from langchain_teddynote.models import MultiModal
from langchain_teddynote.messages import stream_response

llm = ChatOpenAI(
    temperature=0.1,  
    max_tokens=2048,  
    model_name="gpt-4o",  # Multimodal model
)

multimodal_llm = MultiModal(llm)
```

<br>

step 1) 이미지 주소

```python
IMAGE_URL = "https://t3.ftcdn.net/jpg/03/77/33/96/360_F_377339633_Rtv9I77sSmSNcev8bEcnVxTHrXB4nRJ5.jpg"
```

<br>

step 2) 이미지에 대한 해석

```python
answer = multimodal_llm.stream(IMAGE_URL)

stream_response(answer)
```

<br>

### g) Sytem, User 프롬프트 수정

- System prompt: LLM에게 요구하는 **"가치관/정체성"**

- User prompt: LLM에게 던지는 **""구체적인 요구 사항**

```python
system_prompt = """
당신은 표(재무제표) 를 해석하는 금융 AI 어시스턴트 입니다. 
당신의 임무는 주어진 테이블 형식의 재무제표를 바탕으로 흥미로운 사실을 정리하여 친절하게 답변하는 것입니다.
"""

user_prompt = """
당신에게 주어진 표는 회사의 재무제표 입니다. 흥미로운 사실을 정리하여 답변하세요.
"""


multimodal_llm_with_prompt = MultiModal(
  llm, 
  system_prompt=system_prompt, 
  user_prompt=user_prompt
)

```

<br>

이후는 동일

```python
IMAGE_PATH = "https://storage.googleapis.com/static.fastcampus.co.kr/prod/uploads/202212/080345-661/kwon-01.png"

answer = multimodal_llm_with_prompt.stream(IMAGE_PATH)

stream_response(answer)
```

<br>

## (5) LangChain Expression Language (LCEL)

LCEL 3줄 요약

- LangChain에서 ***복잡한 데이터 흐름과 작업***을 ***직관적으로 표현***하기 위해 도입된 언어
- ***데이터의 변환, 필터링, 조건부 실행*** 등을 간결하게 작성할 수 있는 문법을 제공
- 주로 ***프롬프트 템플릿***을 생성하거나, ***체인을 연결***할 때 효율성을 높이는 데 사용

<br>

이번 Section의 task:

- "Prompt - Model - Output parser"를 연결하는 chain 생성할 것

<br>

기본적인 세팅

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging
logging.langsmith("CH01-Basic")
```

<br>

### a) Prompt 템플릿

`PromptTemplate`

- 목적: **"프롬프트 문자열"**을 만들기
  - with 사용자의 입력 변수
- Arguments
  - `template`: 템플릿 문자열 ( 중괄호 `{}`: 변수 )
  - `input_variables`: 변수의 이름을 정의하는 리스트

<br>

Example

```python
from langchain_teddynote.messages import stream_response 
from langchain_core.prompts import PromptTemplate

# template 정의
template = "{country}의 수도는 어디인가요?"

# PromptTemplate 객체 생성
prompt_template = PromptTemplate.from_template(template)
prompt_template
```

```
PromptTemplate(input_variables=['country'], template='{country}의 수도는 어디인가요?')
```

<br>

Prompt 생성하기

```python
# prompt 생성
prompt1 = prompt_template.format(country="대한민국")
prompt2 = prompt_template.format(country="미국")
print(prompt1)
print(prompt2)
```

```
'대한민국의 수도는 어디인가요?'
'미국의 수도는 어디인가요?'
```

<br>

### b) Chain 생성

with **LCEL (LangChain Expression Language)**

- (1) Prompt $$\rightarrow$$ (2) LLM (model) $$\rightarrow$$ (3) Output Parser

<br>

**`|` 기호**: unix 파이프 연산자 ( 서로 다른 구성 요소를 연결 )

```python
chain = prompt | model | output_parser
```

<br>

Example

```python
# (1) Prompt
prompt = PromptTemplate.from_template("{topic} 에 대해 쉽게 설명해주세요.")

# (2) Model
model = ChatOpenAI()

# Chain = (1) | (2) | xxx
chain = prompt | model
```

<br>

### c) `invoke()` , `stream()`

입력값 형식: **python dictionary**

```python
input = {"topic": "인공지능 모델의 학습 원리"}
```



(1) **chain의 invoke 메소드 **

````python
chain.invoke(input)
````

```
AIMessage(content='인공지능 모델의 학습 원리는 데이터를 이용하여 패턴을 학습하는 것입니다. 모델은 입력 데이터를 받아들이고 내부적으로 가중치를 조정하여 원하는 결과를 출력합니다. 학습 과정에서 모델은 입력 데이터와 정답 데이터를 이용하여 오차를 계산하고 이 오차를 최소화하는 방향으로 가중치를 업데이트합니다. 이렇게 반복적으로 학습을 진행하면 모델은 입력 데이터로부터 패턴을 학습하여 정확한 결과를 예측하게 됩니다. 이러한 학습 원리를 통해 인공지능 모델은 데이터를 이용하여 스스로 학습하고 문제를 해결할 수 있습니다.', response_metadata={'token_usage': {'completion_tokens': 214, 'prompt_tokens': 33, 'total_tokens': 247}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-7f8a08f4-51ba-4d14-b9d2-2e092be3e7aa-0', usage_metadata={'input_tokens': 33, 'output_tokens': 214, 'total_tokens': 247})
```

<br>

(2) **chain의 stream 메소드** (for 출력)

```python
answer = chain.stream(input)
stream_response(answer)
```

```
인공지능 모델의 학습 원리는 데이터를 입력으로 받아서 패턴을 학습하고 이를 기반으로 예측이나 분류를 수행하는 과정입니다. 

학습 과정은 크게 입력층, 은닉층, 출력층으로 구성된 인공신경망을 사용합니다. 입력층에서 데이터를 받아 은닉층을 거쳐 출력층으로 결과를 출력하는 구조입니다.

이때, 모델은 주어진 데이터를 통해 가중치를 조정하고 오차를 최소화하는 방향으로 학습을 진행합니다. 이를 위해 주어진 데이터에 대해 예측을 수행하고 실제 값과 비교하여 오차를 계산한 후, 이 오차를 줄이기 위해 가중치를 업데이트합니다.

이러한 반복적인 과정을 통해 모델은 데이터 간의 패턴을 학습하고 새로운 데이터에 대해 정확한 예측을 수행할 수 있게 됩니다. 이렇게 학습된 모델은 새로운 데이터에 대해 일반화된 예측을 할 수 있습니다.
```

<br>

**`invoke ()` vs. `stream()`**

- Invoke

  - **작동 방식**: 주어진 입력에 대해 **"한 번에" 응답을 생성**

  - **결과**: 호출이 완료되면 최종 응답이 반환. 동기적 처리 방식으로, 호출이 끝날 때까지 대기.

  - **사용 사례**: 결과가 빠르게 나오는 경우, 스트리밍 응답이 필요하지 않은 상황

- Stream

  - **작동 방식**:  주어진 입력에 대해 **"스트리밍" 방식으로 응답을 생성**

    ( 응답이 생성되는 동안 데이터가 **실시간**으로 전달 )

  - **결과**:

    - Step 1) `chain.stream(input)`: 스트리밍 응답 객체(`answer`)를 반환

    - Step 2) `stream_response(answer)`: 스트리밍 데이터를 하나씩 출력

  - **사용 사례**: 긴 텍스트 생성, 사용자 대화, 또는 결과를 **실시간으로 출력해야** 하는 상황에 적합

<br>

### d) Output Parser

( chain의 세번째 구성 요소 )

```python
from langchain_core.output_parsers import StrOutputParser

# (3) Output Parser
output_parser = StrOutputParser()

# Chain = (1) | (2) | (3)
chain = prompt | model | output_parser
```

<br>

출력값 형식이 어떻게 달라지는지 확인!

```python
# option 1) invoke
chain.invoke(input)

# option 2) stream
answer = chain.stream(input)
stream_response(answer)
```

```
인공지능 모델의 학습 원리는 데이터를 이용해서 패턴을 학습하는 과정입니다. 먼저 모델은 입력 데이터를 받아서 처리하고, 이때 입력 데이터와 정답 데이터를 비교하여 오차를 계산합니다. 이 오차를 최소화하기 위해 모델은 가중치와 편향을 조정하면서 점차적으로 정확한 패턴을 학습해나갑니다. 이런 과정을 반복하여 모델이 데이터에 대해 정확한 예측을 할 수 있도록 학습시키는 것이 인공지능 모델의 핵심 원리입니다.
```

<br>

### e) Template 변경하여 적용

```python
template = """
당신은 영어를 가르치는 10년차 영어 선생님입니다. 상황에 [FORMAT]에 영어 회화를 작성해 주세요.

상황:
{question}

FORMAT:
- 영어 회화:
- 한글 해석:
"""

# (1) Prompt
prompt = PromptTemplate.from_template(template)

# (2) Model
model = ChatOpenAI(model_name="gpt-4-turbo")

# (3) Parser
output_parser = StrOutputParser()
```

<br>

```python
# Chain = (1)+(2)+(3)
chain = prompt | model | output_parser
```

<br>

```python
answer = chain.stream({"question": 
                       "저는 식당에 가서 음식을 주문하고 싶어요"})

stream_response(answer)
```

```
영어 회화:
- Hello, could I see the menu, please? 
- I'd like to order the grilled salmon and a side of mashed potatoes.
- Could I have a glass of water as well?
- Thank you!

한글 해석:
- 안녕하세요, 메뉴판 좀 볼 수 있을까요?
- 구운 연어와 매시드 포테이토를 주문하고 싶어요.
- 물 한 잔도 주실 수 있나요?
- 감사합니다!
```

<br>

## (6) LCEL 인터페이스

[`Runnable`](https://api.python.langchain.com/en/stable/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable) 프로토콜

- **사용자 정의 체인**을 가능한 쉽게 만들 수 있도록! 대부분의 컴포넌트에 구현되어 있음 
- 표준 인터페이스

  - (a) [`stream`](https://wikidocs.net/233345#stream): 응답의 **청크**를 스트리밍합니다.

  - (b) [`invoke`](https://wikidocs.net/233345#invoke): **입력**에 대해 체인을 호출

  - (c) [`batch`](https://wikidocs.net/233345#batch): **입력 목록**에 대해 체인을 호출
- 비동기 메소드: 앞에 `a`가 붙음

  - [`astream`](https://wikidocs.net/233345#async-stream), [`ainvoke`](https://wikidocs.net/233345#async-invoke), [`abatch`](https://wikidocs.net/233345#async-batch)
  - (d) [`astream_log`](https://wikidocs.net/233345#async-stream-intermediate-steps): (최종 응답뿐만 아니라) 중간 단계를 스트리밍

<br>

**Sync** vs. **Async**

- Sync: 작업이 **"순서대로" 실행**
  - 하나의 작업이 끝날 때까지 기다렸다가, 그 다음 작업을 시작
- Async: 작업이 **"동시"에 실행**
  - 한 작업이 끝날 때까지 기다리지 않고, 다른 작업을 진행

<br>

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging
logging.langsmith("CH01-Basic")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

<br>

Chain 생성

```python
# (1) Prompt
prompt = PromptTemplate.from_template("{topic} 에 대하여 3문장으로 설명해줘.")

# (2) Model
model = ChatOpenAI()

# Chain = (1)+(2)+(3)
chain = prompt | model | StrOutputParser()
```

<br>

아래의 섹션에서는, 다양한 method들에 대해서 알아볼 것!

- stream
- invoke
- batch
- astream (async stream)
- ainvoke (async invoke)
- abatch (async batch)

<br>

### a) stream

데이터 스트림을 생성

( 이 스트림을 반복하여 각 데이터의 내용을 즉시 출력 )

```python
for token in chain.stream({"topic": "멀티모달"}):
    print(token, end="", flush=True)
```

- `end=""`: 줄바꿈 X
- `flush=True`: 출력 buffer 비우기

```
멀티모달은 여러 가지 다른 형태의 커뮤니케이션 수단을 통해 정보를 전달하고 상호작용하는 기술을 의미합니다. 예를 들어 음성, 텍스트, 이미지, 동영상 등 다양한 매체를 활용하여 사용자와 상호작용할 수 있습니다. 멀티모달 기술은 사용자 경험을 향상시키고 정보 전달의 효율성을 높이는데 도움을 줄 수 있습니다.
```

<br>

### b) invoke

해당 주제에 대한 처리를 **한번에** 수행

```
'ChatGPT는 OpenAI에서 개발한 대화형 인공지능 모델로, 다양한 주제에 대한 대화를 자연스럽게 이어나갈 수 있습니다. 사용자들은 ChatGPT를 통해 질문에 답변을 받거나 대화를 이어가며 새로운 정보를 습득할 수 있습니다. 또한 ChatGPT는 사용자의 입력을 학습하여 점차적으로 더욱 유창하고 자연스러운 대화를 제공합니다.'
```

<br>

### c) batch (단위 실행)

**여러 개**의 딕셔너리를 포함하는 리스트를 인자로 받음 (일괄 처리)

```python
chain.batch([{"topic": "ChatGPT"}, 
             {"topic": "Instagram"}])
```

```
['ChatGPT는 인공지능 챗봇으로 자연어 처리 기술을 사용하여 대화를 수행합니다. 사용자들과 자연스럽게 상호작용하며 다양한 주제에 대해 대화할 수 있습니다. ChatGPT는 정보 제공, 질문 응답, 상담 및 엔터테인먼트 등 다양한 용도로 활용될 수 있습니다.', 'Instagram은 사진과 동영상을 공유하고 다른 사람들과 소통하는 소셜 미디어 플랫폼이다. 해시태그를 통해 관심사나 주제별로 사진을 검색하고 팔로워들과 소통할 수 있다. 인기 있는 인플루언서나 브랜드가 활발하게 활동하는 플랫폼으로 세계적으로 인기가 높다.']
```

<br>

argument) `max_concurrency` 

- 동시 요청 수 (처리할 수 있는 최대 작업 수)를 설정

  - **시스템이 한 번에 병렬로 처리할 수 있는 요청의 수**

    ( 순차적으로 처리되는 것이 아니라 동시에 실행되어 작업 속도를 향상시키는 데 활용 )

```python
chain.batch(
    [
        {"topic": "ChatGPT"},
        {"topic": "Instagram"},
        {"topic": "멀티모달"},
        {"topic": "프로그래밍"},
        {"topic": "머신러닝"},
    ],
    config={"max_concurrency": 3},
)
```

<br>

### d) async stream

응답을 비동기적으로 처리

- 비동기 for 루프(`async for`)를 사용
- 스트림에서 메시지를 순차적으로 받아옴

```python
async for token in chain.astream({"topic": "YouTube"}):
    print(token, end="", flush=True)
```

<br>

### e) async invoke

`await` : 비동기로 처리되는 프로세스가 완료될 때까지 기다림

```python
my_process = chain.ainvoke({"topic": "NVDA"})
await my_process
```

<br>

### f) async batch

비동기적으로 일련의 작업을 일괄 처리합니다.

```python
my_abatch_process = chain.abatch(
    [{"topic": "YouTube"}, 
     {"topic": "Instagram"}, 
     {"topic": "Facebook"}]
)

await my_abatch_process
```

<br>

### g) Parallel

LCEL가 병렬 요청을 지원하는 방법?

$$\rightarrow$$ with `RunnableParallel`

<br>

Example:

- Step 1) 주어진 `country`에 대한 **수도** 와 **면적** 을 구하는 두 개의 체인(`chain1`, `chain2`) 생성
- Step 2) `RunnableParallel` 클래스를 사용하여 이 두 체인을 `capital`와 `area`이라는 키로 결합

<br>

```python
from langchain_core.runnables import RunnableParallel

template1 = PromptTemplate.from_template("{country} 의 수도는 어디야?")
template2 = PromptTemplate.from_template("{country} 의 면적은 얼마야?")

chain1 = (template1 | model | StrOutputParser())
chain2 = (template2 | model | StrOutputParser())

combined = RunnableParallel(capital=chain1, area=chain2)
```

<br>

```python
# chain1.invoke({"country": "대한민국"})
# chain2.invoke({"country": "대한민국"})
combined.invoke({"country": "대한민국"})
```

```
{'capital': '대한민국의 수도는 서울입니다.', 'area': '대한민국의 면적은 약 100,363.4 제곱 킬로미터 입니다.'}
```

<br>

(batch) 병렬 처리도 가능!

```python
# chain1.batch([{"country": "대한민국"}, {"country": "미국"}])
# chain2.batch([{"country": "대한민국"}, {"country": "미국"}])
combined.batch([{"country": "대한민국"}, {"country": "미국"}])
```

<br>

## (7) Runnable

(이전과 동일한 세팅)

```python
# 환경 변수 설정
from dotenv import load_dotenv
load_dotenv()

# LangSmith 추적
from langchain_teddynote import logging
logging.langsmith("CH01-Basic")
```

<br>

데이터를 효과적으로 전달하는 방법

- (1) `RunnablePassthrough` 
  - 입력을 변경하지 않거나 추가 키를 더하여 전달
- (2) `RunnablePassthrough()` 
  - 단순히 입력을 받아 그대로 전달
- (3) `RunnablePassthrough.assign()` 
  - assign 함수에 전달할 추가적인 인수를 받음

<br>

### a) RunnablePassthrough

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


# (1) Prompt
prompt = PromptTemplate.from_template("{num} 의 10배는?")

# (2) model
llm = ChatOpenAI(temperature=0)

# chain = (1) + (2)
chain = prompt | llm
```

<br>

Note: `invoke()`의 입력은  **dictionary** 형식이어야 했음

```python
chain.invoke({"num": 5})
```

<br>

(updated version) 변수가 1개일 경우에는 dictionary일 필요 X

- 어차피 입력 변수가 뭔지 알 수 있으니까

```python
chain.invoke(5)
```

<br>

`RunnablePassthrough`

- `runnable` 객체임 $$\rightarrow$$ `invoke()` 메소드 통해 실행 가능

```python
from langchain_core.runnables import RunnablePassthrough
RunnablePassthrough().invoke({"num": 10})
```

```
{'num': 10}
```

<br>

```python
# (Before) chain = (1) prompt + (2) model
# (After) chain = RunnablePassthrough + (1) prompt + (2) model
runnable_chain = {"num": RunnablePassthrough()} | prompt | ChatOpenAI()

runnable_chain.invoke(10)
```

```
AIMessage(content='100입니다.', response_metadata={'token_usage': {'completion_tokens': 3, 'prompt_tokens': 16, 'total_tokens': 19}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-66270ca2-4d62-4c71-859b-de6318f29909-0', usage_metadata={'input_tokens': 16, 'output_tokens': 3, 'total_tokens': 19})
```

<br>

`RunnablePassthrough.assign()`

```python
# v1: RunnablePassthrough
RunnablePassthrough().invoke({"num": 1})
# 출력: {'num': 1}

#---------------------------------------------#
# v2: RunnablePassthrough.assign()
(RunnablePassthrough.assign(new_num=lambda x: x["num"] * 3)).invoke({"num": 1})
# 출력: {'num': 1, 'new_num': 3}
```

<br>

### b) `RunnableParallel`

목적: **여러 Runnable 인스턴스를 병렬로 실행**

```python
from langchain_core.runnables import RunnableParallel

runnable = RunnableParallel(
    passed=RunnablePassthrough(),# 입력 그대로 전달
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    modified=lambda x: x["num"] + 1,
)
```

<br>

출력 결과 예시

```python
runnable.invoke({"num": 1})
```

```
{'passed': {'num': 1}, 'extra': {'num': 1, 'mult': 3}, 'modified': 2}
```

<br>

Chain에도 적용 가능

```python
chain1 = (
    {"country": RunnablePassthrough()}
    | PromptTemplate.from_template("{country} 의 수도는?")
    | ChatOpenAI()
)

chain2 = (
    {"country": RunnablePassthrough()}
    | PromptTemplate.from_template("{country} 의 면적은?")
    | ChatOpenAI()
)
```

```python
combined_chain = RunnableParallel(capital=chain1, 
                                  area=chain2)
combined_chain.invoke("대한민국")
```

```
{'capital': AIMessage(content='서울입니다.', response_metadata={'token_usage': {'completion_tokens': 5, 'prompt_tokens': 19, 'total_tokens': 24}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-d9324c24-9670-4430-97d6-1272f5dbe0f2-0', usage_metadata={'input_tokens': 19, 'output_tokens': 5, 'total_tokens': 24}), 
#---------------------------------------------------#
'area': AIMessage(content='대한민국의 총 면적은 약 100,363 km²입니다.', response_metadata={'token_usage': {'completion_tokens': 24, 'prompt_tokens': 20, 'total_tokens': 44}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-f27442a3-fc9c-4d08-9fdf-189c1b4585c8-0', usage_metadata={'input_tokens': 20, 'output_tokens': 24, 'total_tokens': 44})}
```

<br>

### c) `RunnableLambda`

**사용자 정의 함수** 이용 가능

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from datetime import datetime
```

<br>

ex) 사용자 정의 함수: 오늘 날짜 반환

```python
def get_today(a):
    return datetime.today().strftime("%b-%d")
```

<br>

(Runnable을 포함하여) chain 생성하기

```python
# (1) prompt
prompt = PromptTemplate.from_template(
    "{today} 가 생일인 유명인 {n} 명을 나열하세요. 생년월일을 표기해 주세요."
)

# (2) model
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# (3) parser
# output_parser = StrOutputParser()
```

```python
# chain = Runnable + (1) + (2) + (3) 
chain = (
    {"today": RunnableLambda(get_today), "n": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

<br>

출력 예시

```python
print(chain.invoke(3))
```

```
다음은 6월 19일이 생일인 몇몇 유명인들입니다:

1. 폴 도노반 (Paul Dano) - 1984년 6월 19일
2. 디렉 노박 조코비치 (Novak Djokovic) - 1987년 6월 19일
3. 필리페 쿠티뉴 (Philippe Coutinho) - 1992년 6월 19일

이들은 각각 배우, 테니스 선수, 축구 선수로서 다양한 분야에서 활동하고 있습니다.
```

<br>

특정 딕셔너리의 key값에 해당하는 value 추출

```python
from operator import itemgetter

def get_length(text):
    return len(text)

def _multiple_get_length(text1, text2):
    return len(text1) * len(text2)

def multiple_get_length(_dict):
    return _multiple_get_length(_dict["text1"], 
                                _dict["text2"])
```

<br>

```python
# (1) prompt
prompt = ChatPromptTemplate.from_template("{a} + {b} 는 무엇인가요?")

# (2) model
model = ChatOpenAI()

# chain = Runnable + (1) + (2)
chain = (
    {"a": itemgetter("word1") | RunnableLambda(get_length),
"b": {"text1": itemgetter("word1"), "text2": itemgetter("word2")} | RunnableLambda(multiple_get_length),
    }
    | prompt
    | model
)

chain.invoke({"word1": "hello", "word2": "world"})
```

작동원리

- `"a": itemgetter("word1") | RunnableLambda(get_length)`:
  - step 1-1)  "word1"의 key값을 가지는 value인 "hello"를 반환
  - step 1-2) "hello"의 길이를 계산하여 "5"를 반환
  - 결론: `a` 변수에 5값이 할당
- `"b": {"text1": itemgetter("word1"), "text2": itemgetter("word2")} | RunnableLambda(multiple_get_length)`
  - Dictionary 완성하기
    - step 2) `text1` 변수에 "word1"의 key값을 가지는 value인 "hello"를 할당
    - step 3) `text2` 변수에 "word2"의 key값을 가지는 value인 "world"를 할당
    - 결론: `{text1:"hello", text2:"world"}`
  - `b` 변수에 25이 할당됨 (5x5=25)
- 5 + 25 = 30
