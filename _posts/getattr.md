좋습니다 🙌 이번엔 **Python의 getattr** 에 대해 설명해드릴게요.



------

# `getattr`

- `getattr(object, name[, default])`
  - **객체의 속성 (attribute)** 이름을 **"문자열"**로 받아서 그 속성을 가져오는 함수.
  - 만약 해당 속성이 없으면 **default 값**을 반환할 수 있음 (default 지정 안 하면 AttributeError).

<br>

한 줄 요약: **점(.) 연산자를 동적으로 쓰고 싶을 때** !

```python
value = obj.attr     
value = getattr(obj, "attr")
```



# Example 1: 속성

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

p = Person("Alice", 25)

# 직접 접근
print(p.name)   # "Alice"

# getattr 사용
print(getattr(p, "name"))   # "Alice"
print(getattr(p, "age"))    # 25

# 없는 속성 -> 기본값 반환
print(getattr(p, "height", "Not defined"))   # "Not defined"
```

<br>

# Example 2: 메서드

```python
class Calculator:
    def add(self, a, b):
        return a + b
    def mul(self, a, b):
        return a * b

c = Calculator()
method_name = "mul"
func = getattr(c, method_name)   # c.mul을 가져옴
print(func(3, 4))   # 12
```

<br>

# Example 3: `torch.optim`

## w/o `getattr`

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 1)

optimizer = optim.Adam(model.parameters(), lr=0.001)
```

<br>

## w/ `getattr`

***Optimizer 이름을 문자열로 저장해두고***, `getattr`로 가져오기:

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 1)

# Optimizer 이름을 문자열로 지정
opt_name = "SGD"  # "Adam", "RMSprop" 등 가능

# getattr을 통해 optimizer class 가져오기
OptClass = getattr(optim, opt_name)

# optimizer 생성
optimizer = OptClass(model.parameters(), lr=0.01, momentum=0.9)

print(optimizer)
```

<br>

# Example 4. Config 기반

실제로는 **config(dict)**나 `argparse` **argument**로 Optimizer를 선택할 때 많이 씀:

```python
import torch
import torch.nn as nn
import torch.optim as optim

config = {
    "optimizer": "Adam",
    "lr": 0.001,
    "weight_decay": 1e-4
}

optim_custom = getattr(optim, config["optimizer"])
optimizer = optim_custom(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
```