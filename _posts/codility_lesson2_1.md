



# 2-1. [CyclicRotation](https://app.codility.com/programmers/lessons/2-arrays/cyclic_rotation/)

```python
def solution(A, K):
    N = len(A)
    if N==0:
        return []
    K = K%N
    A1 = A[:N-K]
    A2 = A[N-K:]
    return A2+A1
```

