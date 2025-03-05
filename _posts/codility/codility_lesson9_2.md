# 9-2 (?)

```python
def solution(A):
    import math
    answer = -math.inf
    sum = 0
    for num in A :
        sum = sum + num
        sum = max(num, sum)
        answer = max(answer, sum)
        
    return answer
```

