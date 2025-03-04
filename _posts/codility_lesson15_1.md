# 15-1. [AbsDistinct](https://app.codility.com/programmers/lessons/15-caterpillar_method/abs_distinct/)

```python
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(A):
    A1 = [abs(x) for x in A if x<0]
    A2 = [x for x in A if x>=0]
    A = list(set(A1) | set(A2))
    return len(A)
```

