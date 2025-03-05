# 10-1.

```python
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")
import math
def solution(N):
    n = math.ceil(N**0.5)
    answer = 0
    for i in range(1,n+1):
        if i<=N/i:
            if N%i==0:
                if i==(N//i):
                    answer += 1
                else:
                    answer += 2
    return answer
```

