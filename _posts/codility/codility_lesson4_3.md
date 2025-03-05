# 4-3.

```python
def solution(N, A):
    answer = [0]*N
    max_val = 0
    for x in A:
        if x<=N:
            answer[x-1] +=1
            max_val = max(answer[x-1], max_val)
        else:
            answer = [max_val]*N
    return answer
```

```python

def solution(N, A):
    answer = {key: 0 for key in range(1, N+1)}
    max_val = 0
    for x in A:
        if x<=N:
            answer[x] +=1
            max_val = max(answer[x], max_val)
        else:
            answer = {key: max_val for key in range(1, N+1)}
    
    return list(answer.values())
```

```python
def solution(N, A):
    answer = [0]*N
    max_counter = N+1
    cache = 0
    maximum = 0

    for num in A:
        if num < max_counter:
            if answer[num-1] < cache:
                answer[num-1] = cache + 1
            else:
                answer[num-1] += 1

            if answer[num-1] > maximum:
                maximum = answer[num-1]
        else:
            cache = maximum

    for idx in range(N):
        if answer[idx] < cache:
            answer[idx] = cache

    return answer
```

