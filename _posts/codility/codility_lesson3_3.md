# 3-3.[TapeEquilibrium](https://app.codility.com/programmers/lessons/3-time_complexity/tape_equilibrium/)

```python
def solution(A) :
    
    tape1 = 0
    tape2 = sum(A)
    
    diff = []
    
    for p in A :
        tape1 += p
        tape2 -= p
        
        diff.append(abs(tape1 - tape2))
    
    return min(diff[:-1])
```



