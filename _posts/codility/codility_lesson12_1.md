# 12-1. Chocoloate



```
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(N, M):
    # 0000000000
    # x---x---x-
    # --x---x---
    # y
    # y = x+2N
    # i + 2N = i+M+M+M+M+M
    # i + 2*10 = i+4+4+4+4+4
    # k*N = j*M
    # => M-1
    k = 1000000000
    answer = 0
    for i in range(0,k,M):
        answer += 1
        if i%N==0:
            if i>0:
                return answer-1
```

```
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")
# 
from math import gcd
def solution(N, M):
    # 0000000000
    # x---x---x-
    # --x---x---
    # x---x---x-
    # --x---x

    # y
    # y = x+2N
    # i + 2N = i+M+M+M+M+M
    # i + 2*10 = i+4+4+4+4+4
    # k*N = j*M
    # => M-1
    gcd_ = gcd(N,M)
    N /= gcd_
    M /= gcd_
    
    i = 0
    answer = []
    while True:
        x = (i*M)%N
        if x not in answer:
            answer.append(x)
        else:
            return len(answer)
        i+=1
```

```
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")
from math import gcd

def solution(N, M):
    return N//gcd(N,M)
```

