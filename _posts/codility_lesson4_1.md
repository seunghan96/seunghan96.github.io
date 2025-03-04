# 4-1.

```python
from collections import defaultdict

def solution(X, A):
    cumsum = (X*(X+1))/2
    cnt_dict = defaultdict(lambda: 0)
    for idx,i in enumerate(A):
        if cnt_dict[i]==0:
            cnt_dict[i] = 1
            cumsum -= i
        if cumsum==0:
            return idx
    return -1
```



