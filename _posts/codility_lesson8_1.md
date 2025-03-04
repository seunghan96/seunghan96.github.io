# 8-1.

```python
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")
from collections import Counter

def solution(A):
    N = len(A)
    if N==0:
        return -1
    s = Counter(A)
    result = s.most_common(1)[0]
    max_val = result[0]
    max_cnt = result[1]
    if max_cnt <= N/2:
        return -1
    else:
        return A.index(max_val)
```

