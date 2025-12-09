---
title: Python decorator
categories: [PYTHON]
tags: []
excerpt: property, classmethod, staticmethod


---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# 1. `@property`

한 줄 요약: ***변수처럼 접근***

<br>

## (1) 개념

- Class의 **method를 마치 “변수처럼” 사용할 수 있게 해주는 decorator**
- 즉, method를 호출할 때 `obj.method()`가 아니라 `obj.method`처럼 접근 가능

<br>

## (2) 필요성

- 내부 속성 (private attribute)을 **안전하게 읽도록** 만들기 위해 사용!
- 사용자는 변수처럼 쓰지만, 내부에서는 계산/검증/가공이 가능
- 나중에 구현 변경해도 외부 코드 수정 불필요 (encapsulation)

<br>

## (3) 예시

### a) **변수처럼 보이지만 사실은 계산/검증이 필요한 경우**

@property를 사용 "안할" 경우

```python
class Person:
    def __init__(self, weight, height):
        self.weight = weight
        self.height = height

    def bmi(self):
        return self.weight / (self.height ** 2)

p = Person(70, 1.75)
print(p.bmi())
```

<br>

@property를 사용 "할" 경우

```python
class Person:
    def __init__(self, weight, height):
        self.weight = weight
        self.height = height

    @property
    def bmi(self):
        return self.weight / (self.height ** 2)

p = Person(70, 1.75)
print(p.bmi)   # 변수처럼 접근!
```

<br>

### b) **Class 내부 구현을 바꿔도 외부 코드가 영향을 받지 않음 (Encapsulation)**

예를 들어, "나이(age)"는 해가 지날수록 바뀜

<br>

@property를 사용 "안할" 경우 (== 변수로 정의)

```python
# 기존 정의
user.age

# 필요한 새로운 구현
user.get_age()
```

<br>

@property를 사용 "할" 경우 (== method로 정의)

```python
@property
def age(self):
    return self._birth_year ~ logic...
```

<br>

# 2. `@classmethod`

한 줄 요약: ***class 전체에 영향을 끼침***

<br>

## (1) 개념

- Class 자체를 첫 번째 인자로 받는 method (cls)
- Instance가 아니라 **Class를 다루는 method**

<br>

## (2) 필요성

- **"Class 레벨"에서 동작하는 생성자 추가**하고 싶을 때
- Class 변수를 접근/조작해야 할 때
- 여러 형태의 생성 패턴(팩토리 method)을 만들 때 유용

<br>

classmethod O vs. X

```python
class Korean:
    country = "korea"

    def i_change(self, name):
        self.country = name

    @classmethod
    def c_change(cls, name):
        cls.country = name
```

<br>

변수 확인

```python
a, b = Korean(), Korean()
print(a.country)
print(b.country)
# 출력:
# korea
# korea
```

<br>

Case 1) classmethod (X)

```python
a.i_change("south korea")
print(a.country)
print(b.country)
# 출력:
# south korea
# korea
```

<br>

Case 1) classmethod (O)

```python
a.c_change("south korea")
print(a.country)
print(b.country)
# 출력:
# south korea
# south korea
```

<br>

# 3. **@staticmethod**

한 줄 요약: ***(굳이 불필요한듯) 그냥 함수임***

<br>

## (1) 개념

- Class 내부에 있지만 **Class나 인스턴스와 무관한 ‘독립적인 함수’**
- self나 cls를 받지 않음

<br>

## (2) 필요성

- 논리적으로 Class와 관련된 기능이지만,

  **Class 상태와는 전혀 무관한** util 함수가 필요할 때

- Class 네임스페이스를 깔끔하게 유지하는 목적

<br>

## (3) 예시

```python
class MathUtil:
    @staticmethod
    def add(a, b):
        return a + b

print(MathUtil.add(3, 5))  # 8
```

여기서 add는 Class나 인스턴스 상태랑 무관한 단순 기능이므로 `@staticmethod`가 적합.

<br>

# Summary

| **decorator** | **첫 인자**  | **언제 사용?**                        | **특징**             |
| ------------- | ------------ | ------------------------------------- | -------------------- |
| @property     | self         | 값을 변수처럼 읽되 내부 로직 유지     | Getter 역할          |
| @classmethod  | (관례 상)cls | Class 레벨 동작, 새로운 생성자 만들기 | Class 상태 접근 가능 |
| @staticmethod | 없음         | Class와 관련된 독립 함수              | 독립적 유틸 함수     |

