# 8-2.

```python
from collections import Counter, deque
def solution(A):
    counter = Counter(A)
    result = counter.most_common(1)[0]
    max_val = result[0]
    N = len(A)
    answer = 0
    part1 = []
    part2 = deque(A)
    num_max_part1 = sum([1 for x in part1 if x == max_val])
    num_max_part2 = sum([1 for x in part2 if x == max_val])

    for i in range(N):
        len_part1 = i+1
        len_part2 = N-i-1
        part1.append(A[i])
        part2.popleft()
        if A[i]==max_val:
            num_max_part1+=1
            num_max_part2-=1
        cond1 = (num_max_part1>0.5*len_part1)
        cond2 = (num_max_part2>0.5*len_part2)
        if cond1&cond2:
            answer +=1
    return answer
```



````
# case 1
A = A[1:]

# case 2
A_ = deque(A)
A.popleft()
````

