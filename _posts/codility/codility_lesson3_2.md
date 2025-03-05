# 3-2. [PermMissingElem](https://app.codility.com/programmers/lessons/3-time_complexity/perm_missing_elem/)

```python
def solution(A):
    N = len(A)
    total_sum = ((N+1)*(N+2))/2
    answer = int(total_sum - sum(A))
    return answer
```

