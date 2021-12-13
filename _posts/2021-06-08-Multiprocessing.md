---
title: CPU의 Multiprocessing ( feat. Python )
categories: [PYTHON]
tags: [Python]
excerpt: Multiprocessing 
---

# CPU의 Multiprocessing ( feat. Python )

참고 : [김진휘](https://www.youtube.com/channel/UCoLmMLvQFm5o3aWl-XR8LTQ)

<br>

## 1. Multiprocessing

- CPU의 Multiprocessing
- GPU의 Multiprocessing

<br>

## 2. Python의 Multiprocessing

- Python의 Multiprocessing은 **core 단위가 아닌 thread 단위로 동작**한다

- CPU내에는 여러 개의 core가 존재한다 
  - core 내에는, logical Processor ( = Thread )가 존재한다
  - 일반적으로 core마다 1~2개의 Thread가 존재

<br>

일반적으로 (multiprocessing 없이) 코드를 돌리게 되면,

여러 개의 core 중, 하나의 core만이 동작하게 된다.

<br>

Multiprocessing은, 여러 개의 core를 사용하는 것을 말한다.

각 core에서 연산된 결과들을, 마지막에 모두 merge하여 반환한다.

( python은 이 부분이 취약하여, 성능이 아주 좋지는 않다 )

<br>

## 3. Code

`from multiprocessing import Process`

- `thread=Process(target=함수, args=함수인자)`
- 구조 : `thread.start()` ~ `thread.join()`  

```python
import time,sys
from multiprocessing import Process

def single_core(n):
    result=0
    for i in range(1,n+1):
        result +=i

def multi_core(core_id,num_start,num_end):
    result=0
    for i in range(num_start,num_end+1):
        result +=i
    return

def main(arg=None,num_core=8,n=1000):
    n = int(n)
    num_core = int(num_core)

    if arg=="single":
        start_time=time.time()
        single_core(n)
        end_time=time.time()

    elif arg=="multi":
        tasks = []
        start_time=time.time() 
        for core_id in range(1,num_core+1):
            thread = Process(target=multi_core, args=(core_id,(core_id-1)*n//num_core,(core_id)*n//num_core ))
            tasks.append(thread)
            thread.start()
        for task in tasks:
            task.join()
        end_time=time.time()

    else:
        print('WRONG ARGUMENT!')
        sys.exit()
    
    print(f"Result : {end_time - start_time} sec")

if __name__ == '__main__':
    _val = sys.argv
    main(_val[1],_val[2],_val[3])
```

<br>

```bash
PS C:\Users\LSH\Desktop\advanced_python> python multiprocess_python.py single 8 100000000  
Result : 7.964245557785034 sec

PS C:\Users\LSH\Desktop\advanced_python> python multiprocess_python.py multi 8 100000000  
Result : 2.237269639968872 sec
```

1억을 count하는데에 있어서

- single core 사용 시 : 7.96초
- multi(8) core 사용 시 : 2.24초

***8개를 사용한다고 해서, 그 성능(속도)가 8배나 좋아지는 것은 아니다.***

( 다양한 이유가 있긴 하지만, 대표적으로 core들 끼리 서로 정보 교환하는데에 소요되는 시간 문제도 있다 )

**complete parallelism이 아니다!**