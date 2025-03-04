# 9-1.

```python
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")
import math
def solution(A):
    buy_min = math.inf
    answer = -math.inf
    N = len(A)
    for i in range(1,N):
        sell = A[i]
        buy_min = min(buy_min, A[i-1])
        profit = sell - buy_min
        answer = max(answer, profit)
    if answer<=0:
        return 0
    return answer
```

