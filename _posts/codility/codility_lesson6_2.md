# 6-2.

```python
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(A):
    A = sorted(A,reverse=True)
    A_pos = [x for x in A if x>0]
    A_zero = [x for x in A if x==0]
    A_neg = [x for x in A if x<0]
    
    # Case 1) 양수 3개
    ## 1-1) 양수 3
    ## 1-2) 양수 1 x 음수 2
    if len(A_pos)>=3:
        answer = A_pos[0]*A_pos[1]*A_pos[2]
        if len(A_neg)>=2:
            answer = max(answer, A_pos[0]*A_neg[-1]*A_neg[-2])

    # Case 2) 양수 2개
    ## 2-1) 양수 1 x 음수 2
    ## 2-2) 양수 2 x 0
    ## 2-3) 음수 3
    elif len(A_pos)==2:
        if len(A_neg)>=2:
            answer = A_pos[0]*A_neg[-1]*A_neg[-2]
        elif len(A_neg)==1:
            if len(A_zero)>0:
                answer = 0
            else:
                answer = A[0]*A[1]*A[2]
        else:
            answer = 0

    # Case 3) 양수 1개
    ## 3-1) 양수 1 x 음수 2
    ## 3-2) 양수 1 x 0 x 음수 1
    ## 3-3) 음수 3
    elif len(A_pos)==1:
        if len(A_neg)>=2:
            answer = A_pos[0]*A_neg[-1]*A_neg[-2]
        else:
            answer = 0    

    # Case 4) 양수 0개
    ## 4-1) 0
    ## 4-2) 음수 3개
    elif len(A_pos)==0:
        if len(A_zero)>0:
            answer = 0
        else:
            answer = A[0]*A[1]*A[2]
    return answer
```

