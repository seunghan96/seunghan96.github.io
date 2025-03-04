# 4-4. [MissingInteger](https://app.codility.com/programmers/lessons/4-counting_elements/missing_integer/)

```python
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(A):
    A = sorted(list(set(A)))
    A = [x for x in A if x>0]
    if len(A)==0:
        return 1

    missing = 1
    for x in A:
        if x>0:
            if x==missing:
                missing += 1
            else:
                return missing
    return len(A) + 1
```

