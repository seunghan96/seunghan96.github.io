# 6-3.

```python
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(A):
    answer = 0
    for idx, X in enumerate(A):
        x1 = idx-X
        x2 = idx+X
        for idx2, X2 in enumerate(A[idx+1:]):
            x3 = (idx+idx2+1)-X2
            x4 = (idx+idx2+1)+X2
            if x3<=x2:
                answer += 1

    return answer
```



