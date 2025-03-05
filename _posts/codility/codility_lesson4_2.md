# 4-2.

```py
def solution(A):
    N = len(A)
    required_sum = (N*(N+1))/2
    if len(set(A))!=N:
        return 0
    if sum(A) == required_sum:
        return 1
    else:
        return 0
```

