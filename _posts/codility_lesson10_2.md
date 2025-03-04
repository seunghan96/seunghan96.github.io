# 10-2.

```python
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")
from collections import Counter

def solution(A):
    max_A = max(A)
    answer_dict = dict()
    can_pass = []
    for x in A:
        if x!=max_A:
            if x not in can_pass:
                answer_dict[x] = []
                for y in A:
                    if x%y !=0:
                        answer_dict[x].append(y)
                    elif (x%y ==0)&(x>y):
                        can_pass.append(y)
    can_pass = list(set(can_pass))
    already_keys = list(answer_dict.keys())
    for x in can_pass:
        answer_dict[x] = []
        for k in already_keys:
            if k % x == 0:
                answer_dict[x].extend(answer_dict[k])
    
    answer_dict[max_A] = []
    answer = [] 
    for k,v in answer_dict:
        if k in 
    
            





```

