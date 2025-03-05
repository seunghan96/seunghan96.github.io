# 2-2. [OddOccurrencesInArray](https://app.codility.com/programmers/lessons/2-arrays/odd_occurrences_in_array/)

```python
from collections import Counter

def solution(A):
    counter = Counter(A)
    for k, v in counter.items():
        if v%2==1:
            answer= k
    return answer
```

