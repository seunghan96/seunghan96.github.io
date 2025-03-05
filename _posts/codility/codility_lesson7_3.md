# 7-3. [Nesting](https://app.codility.com/programmers/lessons/7-stacks_and_queues/nesting/)

```python
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(S):
    add_list = []
    for s in S:
        if s=='(':
            add_list.append(s)
        else:
            if len(add_list)==0:
                return 0
            else:
                add_list.pop()
    if len(add_list)==0:
        return 1
    else:
        return 0
        
```

