# 5-2. [CountDiv](https://app.codility.com/programmers/lessons/5-prefix_sums/count_div/)

```python
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(A, B, K):
    a1 = A//K
    a2 = A%K
    b1 = B//K
    b2 = B%K
    answer = b1 - a1
    if a2==0:
        answer += 1
    return answer
```







