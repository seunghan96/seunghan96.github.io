# 5-1. [PassingCars](https://app.codility.com/programmers/lessons/5-prefix_sums/passing_cars/)

```python
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(A):
    zero_list = [idx for idx,x in enumerate(A) if x==0]
    total_sum = 0
    for z in zero_list:
        total_sum += sum(A[z+1:])
    if total_sum>1000000000:
        return -1
    return total_sum
```

```python
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(A):
    A = list(map(str, A))
    A = ''.join(A)
    A = A.split('0')
    A_len = [len(x) for x in A]
    cumsum=0
    for idx,x in enumerate(A_len):
        cumsum += x*idx
        if cumsum>1000000000:
            return -1
    return cumsum
    
```

