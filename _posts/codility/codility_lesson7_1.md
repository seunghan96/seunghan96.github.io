# 7-1.

```python
def solution(S):
    dict_ = {"(":")", "{":"}","[":"]"}
    open_ = list(dict_.keys())
    x = []

    for s in S:
        if s in open_:
            x.append(s)
        else:
            if (len(x)==0) or dict_[x.pop()]!=s:
                return 0
    
    if len(x)>0:
        return 0
    else:
        return 1
    
```

