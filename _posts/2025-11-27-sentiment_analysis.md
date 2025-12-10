---
title: (OpenAI API) Sentiment Analysis
categories: [LLM]
tags: []
excerpt: 
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# (OpenAI API) Sentiment Analysis

## Contents

1. Sentiment Analysis with OpenAI + Reddit 
2. Python Docstring 작성하기
3. Translation

<br>

# 1. Sentiment Analysis with OpenAI + Reddit (`praw`)

<br>

## (1) Project

1. `praw` (Python Reddit API Wrapper)로 Reddit 댓글/게시글을 가져오고
2. OpenAI API에 텍스트를 보내서
3. **감성 분석 결과 (positive / negative / neutral + 이유)** 를 받는 파이프라인

<br>

## (2) 기본 아이디어 (한 줄 요약)

- Step 1) `praw`로 Reddit에서 텍스트 가져오기

- Step 2) OpenAI `responses.create()`로 감성 분석 요청

- Step 3) 결과(JSON) 파싱해서 사용.

<br>

## (3) 예시 코드 

```python
from openai import OpenAI
import praw
import os

# OpenAI 클라이언트 
# 참고) 사전에 환경변수 OPENAI_API_KEY 필요)
client = OpenAI()

# Reddit 클라이언트
# 참고) https://www.reddit.com/prefs/apps 에서 발급
reddit = praw.Reddit(
    client_id=os.environ["REDDIT_CLIENT_ID"],
    client_secret=os.environ["REDDIT_CLIENT_SECRET"],
    user_agent="sentiment-analysis"
)
```

<br>

하나의 텍스트에 대해 sentiment 분석 

- structured output 활용

```python
def analyze_sentiment(text: str):
    """하나의 텍스트에 대해 sentiment 분석 (structured output 활용)"""
    
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=f"Analyze the sentiment of the following text:\n\n{text}",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "SentimentResult",
                "schema": {
                    "type": "object",
                    "properties": {
                        "sentiment": {
                            "type": "string",
                            "enum": ["positive", "neutral", "negative"]
                        },
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"}
                    },
                    "required": ["sentiment", "reason"],
                    "additionalProperties": False
                }
            }
        }
    )
    
    # 구조화된 JSON 그대로 반환
    content = response.output[0].content[0].text
    
    # text는 이미 JSON 문자열 형태이므로, 필요하면 json.loads(content) 사용
    return content
```

<br>

Example) 특정 subreddit에서 상위 댓글 몇 개 긁어오기

```python
# (1) 주제
subreddit_name = "finance"

# (2) 가장 hot한 글 Top 3
top_K = 3

# (3) 게시글(submission) 불러오기
submission = next(reddit.subreddit(subreddit_name).hot(limit=top_K))  

print(f"Title: {submission.title}\n")
submission.comments.replace_more(limit=0)
for i, comment in enumerate(submission.comments[:5]):
    print(f"\n[Comment {i}] {comment.body[:200]}...\n")
    result_json = analyze_sentiment(comment.body)
    print("Sentiment result:", result_json)
```

<br>

코드 Summary

- `praw.Reddit(...)`로 Reddit 세션 생성
- `submission.comments` 로 댓글 가져옴
- `analyze_sentiment()`에서 OpenAI **Structured Output** 사용 → 항상 다음 같은 JSON 구조 보장!

```json
{
  "sentiment": "positive",
  "confidence": 0.87,
  "reason": "The user expresses satisfaction and gratitude."
}
```

<br>

# 2. Python Docstring 작성하기

## (1) Python Docstring 기본 세 가지

### **1) One-line docstring**

```
def add(a, b):
    """Return the sum of a and b."""
    return a + b
```

<br>

### **2) Multi-line docstring**

```
def normalize(x, eps=1e-8):
    """
    Normalize a 1D array to have zero mean and unit variance.

    Parameters
    ----------
    x : np.ndarray
        Input 1D array.
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Normalized array.
    """
    ...
```

<br>

### **3) reStructuredText (reST) docstrings**

- Sphinx 같은 문서화 툴에서 **자동 문서 생성**할 때 자주 쓰는 포맷
- :param, :type, :return:, :rtype: 같은 태그 사용

```
def sigmoid(x):
    """
    Compute the sigmoid of x.

    :param x: Input value or array.
    :type x: float or np.ndarray
    :return: Sigmoid of x.
    :rtype: float or np.ndarray
    """
    ...
```

특징:

- **기계가 파싱하기 좋은 형식**이라 문서 생성 툴과 궁합이 좋음!
- OpenAI로 자동 생성 시에도 “이 포맷으로 작성해줘”라고 지정하기 좋음

<br>

## (2) `__doc__`, `help()` , `inspect.getsource()` 함수

a) `__doc__` 함수

```python
def add(a, b):
    """Return the sum of a and b."""
    return a + b

print(add.__doc__)
# "Return the sum of a and b."
```

<br>

b) `help()` 함수

- docstring을 예쁘게 포맷해서 보여줌
- 자동 생성한 docstring이 잘 표시되는지 QA할 때 한 번씩 써보기 좋음

```python
help(add)
```

```
Help on function add in module __main__:

add(a, b)
    Return the sum of a and b.
```

<br>

c) `inspect.getsource(my_function)`

- inspect 모듈의 기능
- 함수/클래스/모듈의 **소스 코드를 문자열로 가져옴**

```python
import inspect

def foo(x):
    """Example."""
    return x + 1

print(inspect.getsource(foo))
```

출력:

```
def foo(x):
    """Example."""
    return x + 1
```

<br>

중요한 이유?

- **OpenAI로 docstring을 자동 생성할 때,** 함수 정의부 + 본문 코드를 통째로 모델에 넣기 위해 활용

- e.g., *“이 함수 소스가 여기 있으니, 적절한 reST docstring을 만들어줘.”*

<br>

## (3) “Python Docstring” 자동 작성하기

## (1) Workflow

1. Python 코드에서 함수/클래스를 `inspect`로 가져오고
2. `inspect.getsource()`로 소스 코드를 문자열로 뽑은 뒤
3. OpenAI API에 *“이 함수에 대한 docstring을 작성해줘”*라고 요청
4. (reST / NumPy / Google 스타일 등) 원하는 format을 Prompt에 명시
5. 생성된 docstring을 코드에 붙여넣거나, 리뷰 후 반영

<br>

## (2) Example

```python
from openai import OpenAI
import inspect

client = OpenAI()

def compute_scores(preds, labels, eps=1e-8):
    scores = []
    for p, y in zip(preds, labels):
        if p + y == 0:
            scores.append(0.0)
        else:
            scores.append(2 * p * y / (p + y + eps))
    return sum(scores) / len(scores)


def generate_docstring(func):
    source = inspect.getsource(func)

    prompt = f"""
You are an expert Python developer.

Write a reStructuredText (reST) style Python docstring for the following function.
Follow these rules:
- First line: short summary (one sentence).
- Then a blank line.
- Then detail with :param, :type, :return, :rtype:
- Do NOT modify the function code.
- Only output the docstring content, without the triple quotes.

Function source:
{source}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    docstring_text = response.output[0].content[0].text.strip()
    return docstring_text


if __name__ == "__main__":
    doc = generate_docstring(compute_scores)
    print("Generated docstring:\n")
    print('"""')
    print(doc)
    print('"""')
```

예상 출력(예시):

```clike
"""
Compute the average symmetric score between predictions and labels.

:param preds: Iterable of prediction values.
:type preds: list[float] or np.ndarray
:param labels: Iterable of ground-truth label values.
:type labels: list[float] or np.ndarray
:param eps: Small constant to avoid division by zero.
:type eps: float
:return: Average symmetric score over all prediction-label pairs.
:rtype: float
"""
```

<br>

# 3. Translation

```python
from openai import OpenAI

client = OpenAI()

text = "이 문장을 영어로 번역해줘."

response = client.responses.create(
    model="gpt-4.1-mini",
    input=f"Translate the following into English:\n\n{text}"
)

print(response.output[0].content[0].text)
```

출력 예:

```
Please translate this sentence into English.
```

