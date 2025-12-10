```
# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(client):
    N = len(client) # number of clients (packages)
    current_idx = 0 # current client index
    on_shelf = [False] * (N+1) # package on the shelf (X/O)
    shelf = 0 # number of packages on the shelf
    answer = 0 
     
    for p in range(1, N+1):
        if (current_idx < N) and (client[current_idx] == p):
            current_idx += 1
            while (current_idx < N) and (on_shelf[client[current_idx]]):
                on_shelf[client[current_idx]] = False
                shelf -= 1
                current_idx += 1
        else:
            on_shelf[p] = True
            shelf += 1
            if shelf > answer:
                answer = shelf
            while (current_idx < N) and (on_shelf[client[current_idx]]):
                on_shelf[client[current_idx]] = False
                shelf -= 1
                current_idx += 1
    return answer

```

