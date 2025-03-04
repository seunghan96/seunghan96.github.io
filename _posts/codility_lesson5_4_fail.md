# 5-4.

```python
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")
import math
def solution(A):
    N = len(A)
    answer_list = []
    min_val = math.inf
    for l in range(2,N):
        temp = []
        for k in range(l):
            temp.append(A[0:k]+A+[999]*(N-k))
        result = list(map(lambda x: sum(x) / len(x), zip(*temp)))
        result = result[l-1:N]
        #print(result)
        
        for idx, x in enumerate(result):
            if x < min_val:
                min_val = x
                answer_list = []
                answer_list.append(idx+l-2)
            elif x==min_val:
                answer_list.append(idx+l-2)
    return min(answer_list)

  
```



