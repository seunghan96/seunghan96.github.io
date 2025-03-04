# 16-2. [TieRopes](https://app.codility.com/programmers/lessons/16-greedy_algorithms/tie_ropes/)

```python
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(K, A):
    num_ropes = 0
    length_rope = 0
    for rope in A:
        length_rope += rope
        if length_rope >=K:
            num_ropes +=1
            length_rope = 0
    return num_ropes


```

