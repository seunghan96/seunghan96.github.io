# 16-1. [MaxNonoverlappingSegments](https://app.codility.com/programmers/lessons/16-greedy_algorithms/max_nonoverlapping_segments/)

```python
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(A, B):
    N = len(A)
    if (len(A)==0)|(len(B)==0):
        return 0
    if N==1:
        return 1

    answer = 1
    endpoint = B[0]
    for i in range(1,N):
        startpoint = A[i]
        if startpoint > endpoint:
            endpoint = B[i]
            answer +=1
    return answer
```

