---
title: Langchain 실습 2
categories: [LLM, NLP]
tags: []
excerpt: -

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Langchain 실습 1

# 2. Prompt

## (0) Intro

Prompt = LLM에 던지는 질문/명령

원하는 답변을 얻기 위해 필수적인 단계! 

<br>

### a) 프롬프트의 필요성

- (1) **문맥(Context) 설정**: LLM이 특정 문맥 하에서 작동하도록 설정
- (2) **정보 통합**: 여러 문서에서 검색된 정보는 서로 다른 관점이나 내용을 포함. 이러한 내용을 전부 담도록
- (3) **응답 품질 향상**: 프롬프트에 따라 응답 품질이 크게 영향 받음

<br>

### b) RAG 프롬프트 구조

- (1) 지시 사항 (Instruction)
- (2) 질문 (사용자 입력 질문)
- (3) 문맥 (검색된 정보)

```yaml
# (1) 지시 사항
당신은 질문-답변(Question-Answer) Task 를 수행한는 AI 어시스턴트 입니다.
검색된 문맥(context)를 사용하여 질문(question)에 답하세요. 
만약, 문맥(context) 으로부터 답을 찾을 수 없다면 '모른다' 고 말하세요. 
한국어로 대답하세요.

# (2) 질문
#Question: 
{이곳에 사용자가 입력한 질문이 삽입됩니다}

# (3) 문맥
#Context: 
{이곳에 검색된 정보가 삽입됩니다}
```

<br>

## (1) 프롬프트

### a) `PromptTemplate`

기본적인 설정

```python
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI

load_dotenv()

logging.langsmith("CH02-Prompt")
llm = ChatOpenAI()
```

<br>

**방법 1)  `from_template()` 사용 (O)**

```python
from langchain_core.prompts import PromptTemplate

template = "{country}의 수도는 어디인가요?"
prompt = PromptTemplate.from_template(template)
```

<br>

```python
prompt_ex1 = prompt.format(country="대한민국")
prompt_ex1
```

```
'대한민국의 수도는 어디인가요?'
```

<br>

```python
chain = prompt | llm
output_ex1 = chain.invoke("대한민국").content
print(output_ex1)
```

```
'대한민국의 수도는 서울입니다.'
```

<br>

**방법 2)  `from_template()` 사용 (X)**

- (객체 생성과 동시에) 프롬프트 생성
- default 입력값은 `input_variables`에 넣은 변수이다!

```python
template = "{country}의 수도는 어디인가요?"

prompt = PromptTemplate(
    template=template,
    input_variables=["country"],
)
```

<br>

```python
prompt_ex2 = prompt.format(country="대한민국")
prompt_ex2
```

```
'대한민국의 수도는 어디인가요?'
```

<br>

**심화** : 2개의 입력 변수

```python
template = "{country1}과 {country2}의 수도는 각각 어디인가요?"

prompt = PromptTemplate(
    template=template,
    input_variables=["country1"],
    partial_variables={
        "country2": "미국"  
    },
)
```

<br>

```python
prompt_ex3 = prompt.format(country1="대한민국")
prompt_ex3
```

```
'대한민국과 미국의 수도는 각각 어디인가요?'
```

<br>

```python
prompt2 = prompt.partial(country2="캐나다")
prompt_ex4 = prompt2.format(country1="대한민국")
prompt_ex4
```

```
'대한민국과 캐나다의 수도는 각각 어디인가요?'
```

<br>

```python
chain = prompt_partial | llm
output_ex2a = chain.invoke("대한민국").content
output_ex2b = chain.invoke({"country1": "대한민국", "country2": "호주"}).content

print(output_ex2a)
print(output_ex2b)
```

```
'대한민국의 수도는 서울이며, 캐나다의 수도는 오타와입니다.'
'대한민국의 수도는 서울이고 호주의 수도는 캔버라입니다.'
```

<br>

### b) `partial_variables`

- 목적: 함수를 **부분적으로 사용**

  - When? **항상 공통된 방식으로 가져오고 싶은 변수** 가 있는 경우

    ( e.g., 날짜나 시간 )

<br>

현재 날짜가 항상 표시되기를 원하는 프롬프트

```python
from datetime import datetime

def get_today():
    return datetime.now().strftime("%B %d")
```

```python
# (1) prompt
prompt = PromptTemplate(
    template="오늘의 날짜는 {today} 입니다. 오늘이 생일인 유명인 {n}명을 나열해 주세요. 생년월일을 표기해주세요.",
    input_variables=["n"],
    partial_variables={
        "today": get_today  
    },
)

# (2) model
llm = ChatOpenAI()

# chain = (1) + (2)
chain = prompt | llm
```

<br>

(Default: 오늘의 날짜를 불러와서 응답 생성)

```python
print(chain.invoke(3).content)
```

```
1. Nicole Kidman - 1967년 6월 20일
2. John Goodman - 1952년 6월 20일
3. Lionel Richie - 1949년 6월 20일
```

<br>

지정해서도 가능!

```python
print(chain.invoke({"today": "Jan 02", "n": 3}).content)
```

```
1. Kate Bosworth - 1983년 1월 2일
2. Tia Carrere - 1967년 1월 2일
3. Christy Turlington - 1969년 1월 2일
```

<br>

### c) 파일로부터 template 읽어오기

```python
from langchain_core.prompts import load_prompt

prompt = load_prompt("prompts/fruit_color.yaml")
prompt
```

```
PromptTemplate(input_variables=['fruit'], template='{fruit}의 색깔이 뭐야?')
```

<br>

```python
prompt_ex1 = prompt.format(fruit="사과")
print(prompt_ex1)
```

```
'사과의 색깔이 뭐야?'
```

<br>

### d) `ChatPromptTemplate`

- 목적: **"대화 목록"** 을 프롬프트로 주입하고자 할 때
- 형식: **튜플 (tuple)**
  - (`role`, `message`) 로 구성하여 리스트로 생성
- 세부 사항
  - `"system"`: 시스템 설정 메시지
  - `"human"` : 사용자 입력 메시지
  - `"ai"`: AI 의 답변 메시지

<br>

Ex 1) `from_template()`

```python
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_template("{country}의 수도는 어디인가요?")

prompt_ex1 = chat_prompt.format(country="대한민국")
prompt_ex1
```

```
'Human: 대한민국의 수도는 어디인가요?'
```

<br>

Ex 1) `from_messages()`

```python
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 친절한 AI 어시스턴트입니다. 당신의 이름은 {name} 입니다."),
        ("human", "반가워요!"),
        ("ai", "안녕하세요! 무엇을 도와드릴까요?"),
        ("human", "{user_input}"),
    ]
)

messages = chat_template.format_messages(
    name="테디", user_input="당신의 이름은 무엇입니까?"
)
messages
```

```
[SystemMessage(content='당신은 친절한 AI 어시스턴트입니다. 당신의 이름은 테디 입니다.'), HumanMessage(content='반가워요!'), AIMessage(content='안녕하세요! 무엇을 도와드릴까요?'), HumanMessage(content='당신의 이름은 무엇입니까?')]
```

<br>

```python
output_ex1 = llm.invoke(messages).content
print(output_ex1)
```

```
'제 이름은 테디입니다. 필요한 도움이 있으면 언제든지 말씀해주세요!'
```

<br>

Chain 생성

```python
chain = chat_template | llm
output_ex2 = chain.invoke({"name": "Teddy",
                           "user_input": "당신의 이름은 무엇입니까?"}).content
print(output_ex2)
```

```
'제 이름은 Teddy입니다. 어떻게 도와드릴까요?'
```

<br>

### e) `MessagePlaceholder`

- 목적: 렌더링할 **메시지를 완전히 제어**하기 위해
- When? **메시지 프롬프트 템플릿에 어떤 역할을 사용해야 할지 확실하지 않거나 서식 지정 중에 메시지 목록을 삽입하려는 경우**

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","당신은 요약 전문 AI 어시스턴트입니다. 당신의 임무는 주요 키워드로 대화를 요약하는 것입니다."),
      
      
      MessagesPlaceholder(variable_name="conversation"),
      
      
        ("human", "지금까지의 대화를 {word_count} 단어로 요약합니다."),
    ]
)
```

<br>

위와 같이 작성 할 경우,  `conversation` 대화목록을 **나중에/원할 때** 추가 가능!

```python
formatted_chat_prompt = chat_prompt.format(
    word_count=5,
    conversation=[
        ("human", "안녕하세요! 저는 오늘 새로 입사한 테디 입니다. 만나서 반갑습니다."),
        ("ai", "반가워요! 앞으로 잘 부탁 드립니다."),
    ],
)

print(formatted_chat_prompt)
```

```
System: 당신은 요약 전문 AI 어시스턴트입니다. 당신의 임무는 주요 키워드로 대화를 요약하는 것입니다.
Human: 안녕하세요! 저는 오늘 새로 입사한 테디 입니다. 만나서 반갑습니다.
AI: 반가워요! 앞으로 잘 부탁 드립니다.
Human: 지금까지의 대화를 5 단어로 요약합니다.
```

<br>

Chain 생성

```python
chain = chat_prompt | llm | StrOutputParser()


chain.invoke(
    {
        "word_count": 5,
        "conversation": [
            ("human","안녕하세요! 저는 오늘 새로 입사한 테디 입니다. 만나서 반갑습니다."),
            ("ai", "반가워요! 앞으로 잘 부탁 드립니다."),
        ],
    }
)
```

```
# 5 단어로 대화가 요약된 것을 확인할 수 있음!
'새로운 입사자 테디 만남.'
```

<br>

## (2) Few-shot Prompt ( `FewShotPromptTemplate` )

기본 설정

```python
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import stream_response

load_dotenv()

logging.langsmith("CH02-Prompt")

llm = ChatOpenAI(
    temperature=0,  
    model_name="gpt-4-turbo",
)
```

<br>

```python
question = "대한민국의 수도는 뭐야?"
answer = llm.stream(question)
stream_response(answer)
```

```
대한민국의 수도는 서울입니다.
```

<br>

간단한 example

```python
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

fewshot_ex = examples = [
    {"input": "고양이", "output": "Cat"},
    {"input": "강아지", "output": "Dog"}
]

template_ex = "Input: {input}\nOutput: {output}"
example_prompt = FewShotPromptTemplate.from_examples(
    examples=fewshot_ex,
    example_prompt_template=template_ex,
    input_variables=["input"],
    suffix="Input: {input}\nOutput:", 
)
```

<br>

Few-shot 예시들과 함께 출력된 것을 확인할 수 있다.

```python
final_prompt = example_prompt.format(input="토끼")
print(final_prompt)
```

```
Input: 고양이
Output: Cat

Input: 강아지
Output: Dog

Input: 토끼
Output:
```

<br>

## (3) LangChain Hub

### a) Hub로부터 Prompt 받아오기

Prompt를 **LangChain Hub**에서도 받아올 수 있음

다양한 방법

- (1) 프롬프트 repo의 아이디 값을 가져오기
- (2) commit id 를 붙여서 특정 버전에 대한 프롬프트를 받아오기

```python
from langchain import hub

prompt = hub.pull("rlm/rag-prompt")
print(prompt)
```

```
input_variables=['context', 'question'] metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'} messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"))]
```

<br>

```python
prompt = hub.pull("rlm/rag-prompt:50442af1")
prompt
```

```
ChatPromptTemplate(input_variables=['context', 'question'], metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"))])
```

<br>

### Hub에 프롬프트 등록하기

```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "주어진 내용을 바탕으로 다음 문장을 요약하세요. 답변은 반드시 한글로 작성하세요\n\nCONTEXT: {context}\n\nSUMMARY:"
)

hub.push("teddynote/simple-summary-korean", prompt)

# 잘 불러와지는지 검토
# pulled_prompt = hub.pull("teddynote/simple-summary-korean")
```

<br>



