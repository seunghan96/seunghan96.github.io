# 3-1. [FrogJmp](https://app.codility.com/programmers/lessons/3-time_complexity/frog_jmp/)

```python
def solution(X, Y, D):
    if (Y-X)%D==0:
        answer = (Y-X)//D
    else:
        answer = (Y-X)//D + 1
    
    return answer
```

