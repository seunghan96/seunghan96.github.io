```
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(S):
    N = len(S)
    answer = 0

    x_start = 0
    y_start = 0
    x_list = [0] * (N+1)
    y_list = [0] * (N+1)

    for idx in range(N-1,-1,-1):
        is_x = S[idx] == 'x'
        is_y = S[idx] == 'y'
        x_list[idx] = x_list[idx+1] + is_x
        y_list[idx] = y_list[idx+1] + is_y
    
    for idx in range(1, N):
        is_x = S[idx-1] == 'x'
        is_y = S[idx-1] == 'y'
        x_start += int(is_x)
        y_start += int(is_y)
        
        if (x_start == y_start) or (x_list[idx] == y_list[idx]):
            answer += 1
    
    return answer



    
```

