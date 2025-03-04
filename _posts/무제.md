

# Lesson 1. [BinaryGap](https://app.codility.com/programmers/lessons/1-iterations/binary_gap/)

```
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(N):
    # (1) Find the length of string
    N2 = N
    len_ = 0
    while True:
        len_ +=1
        N2 /=2
        if N2<1:
            break
    
    # (2) Insert 1,0
    answer = []
    for i in range(len_-1,-1,-1):
        temp = N - (2**i)
        if temp>=0:
            answer.append('1')
            N = temp
        else:
            answer.append('0')
    answer = ''.join(answer)
    answer = answer.split('1')
    answer = answer[:-1]
    answer = [len(x) for x in answer]
    return max(answer)

```



