# 12-3.

```python
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

import math
from collections import defaultdict

def is_check_prime(n, dict_): 
    if n==1:
        dict_[n] = -1
        return False
    for i in range(2, n):
        if n%i == 0: 
            dict_[n] = -1
            return False
    dict_[n] = 1
    return True

def is_prime(n, dict_): 
    if dict_[n]==1:
        return True
    elif dict_[n]==-1:
        return False
    else:
        if is_check_prime(n, dict_):
            return True
        else:
            return False

def get_divisor(N, dict_):
    answer = []
    from_ = 1
    to_ = int(N**0.5)+1
    for i in range(from_,to_):
        if N%i==0: 
            if is_prime(i, dict_):
                answer.append(i)
            if is_prime(int(N//i), dict_):
                answer.append(int(N//i))
    return answer, dict_

def solution(A, B):
    answer = 0
    is_prime_dict = defaultdict(lambda:0)
    for idx,(a,b) in enumerate(zip(A,B)):
        a_divisor, is_prime_dict = get_divisor(a, is_prime_dict)
        b_divisor, is_prime_dict = get_divisor(b, is_prime_dict)
        if b_divisor==a_divisor:
            answer += 1
    return answer

    
```

