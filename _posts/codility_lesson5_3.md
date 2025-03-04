# 5-3. [GenomicRangeQuery](https://app.codility.com/programmers/lessons/5-prefix_sums/genomic_range_query/)

```python
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(S, P, Q):
    order_ascending = ['A','C','G','T']
    answer = [] 
    for p, q in zip(P,Q):
        x = S[p:q+1]
        for idx, a in enumerate(order_ascending):
            if a in x:
                answer.append(idx+1)
                break
    return answer
```

<br>

