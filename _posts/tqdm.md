# [ TQDM ]

- 작업 진행 progress bar를 확인할 수 있게 해주는 패키지

<br>

```python
from tqdm import tqdm,trange
import time 

def main():
    # range
    for i in tqdm(range(10)) :
        time.sleep(0.25) 
    
    # trange
    for i in trange(3):
        time.sleep(0.25)

    # list 형식
    text=''
    for i in tqdm(['a','b','c','d']) :
        time.sleep(0.5) 
        text=text+i
    print(text)
    
if __name__ == "__main__":
    main()
```

<br>

Output

```
PS C:\Users\LSH\Desktop\python_basic> python test.py
100%|███████████████████████████████████████████████████████████████| 10/10 [00:02<00:00,  3.92it/s]
100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.87it/s]
100%|█████████████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.97it/s]
abcd
```

