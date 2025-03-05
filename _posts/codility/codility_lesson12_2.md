# 12-2.

```python
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

import math

def get_divisor(N):
    answer = []
    from_ = 1
    to_ = int(N**0.5)+1
    for i in range(from_,to_):
        if N%i==0: 
            answer.append(i)
            answer.append(int(N//i))
    return answer

def solution(A, B):
    for idx,(a,b) in enumerate(zip(A,B)):
        a_divisor = get_divisor(a)
        b_divisor = get_divisor(b)
        print(a_divisor)
        print(b_divisor)
        break

    
```

