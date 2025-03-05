# 15-2.

```python
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(M, A):
    num_slices = 0
    N = len(A)
    start_idx = 0
    while start_idx < N:
        add_idx = 0
        temp = []
        temp.append(A[start_idx])
        # (0) = (0) 1->1 (1*2)/2 = 1
        #--------------------------------------------#
        # (0,1) =(0) + (0,1) 2->2 (2*3)/2 = 3
        # (0,2) = (0) + (1) + (0,1) + (0,2) + (1,2) # 3->5 
        # (0,3) = 0/1/2/01/02/03/12/13/23/ # 4->9
        for n in range(start_idx+1,N):
            if A[n] in temp:
                break
            else:
                add_idx += 1
                temp.append(A[n])
        if len(temp)==1:
            add_idx += 1
            num_slices += 1
        else:
            num_slices += int((len(temp)*(len(temp)+1))/2)-1
        if start_idx+1 == N:
           break
        start_idx += add_idx
         
        
    return num_slices
```

