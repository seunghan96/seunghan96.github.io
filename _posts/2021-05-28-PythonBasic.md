---
title: Useful Python Libraries
categories: [PYTHON]
tags: [Python]
excerpt: Import, Argparse, Logging, os, tqdm, csv & pickle
---

# Useful Python Libraries

1. Import
2. Argparse
3. Logging
4. os
5. tqdm
6. csv & pickle

<br>

# 1. Import

현재 경로 상에 있는 ***"myfolder"***폴더, 그 안에 ***"mypy.py"*** 파이썬 파일

```python
def my_add(x,y):
    return(x+y)

def my_sub(x,y):
    return(x-y)

class my_Class:
    def __init__(self,x,y):
        self.x=x
        self.y=y
    
    def my_Class_add(self,multiplier):
        return multiplier*(self.x+self.y)
```

<br>

### Example 1

```python
from myfolder.mypy import my_add

x=1
y=2
print(my_add(x,y))
```

Output :

```
3
```

<br>

### Example 2

```python
from myfolder.mypy import *

x=1
y=2
print(my_add(x,y))
print(my_sub(x,y))
```

Output :

```
3
-1
```

<br>

### Example 3

```python
#from myfolder.mypy import *
from myfolder.mypy import my_Class

x=1
y=2
mult=0.01
c = my_Class(x,y)

print(c.my_Class_add(mult))
```

Output :

```
0.03
```

<br>

# 2. Argparse

***terminal에 python 파일 실행 시, "--"뒤에 argument를 입력하는 패키지***

<br>

필수 코드

- `parser = argparse.ArgumentParser()`
- `parser.add_argument("--data",~~~)`
- `args = parser.parse_args()`

<br>

추가 사항

- `choices` : 해당 list안에 있는 값들 중 하나로 입력을 해야! ( o.w error )
- `required=True` : 반드시 입력을 줘야함

<br>

입력한 argument 값:

- `args.data`

```python
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x",
                        type=float,
                        default=1.0,
                        help="What is x?")
    parser.add_argument("--y",
                        type=str,
                        default='ABC',
                        choices=["ABC", "DEF"],
                        help="What is y?")
    parser.add_argument("--z",
                        type=int,
                        default=5,
                        required=True,
                        help="What is z?")
    args = parser.parse_args()
    print('x = ',args.x)
    print('y = ',args.y)
    print('z = ',args.z)
    
if __name__ == "__main__":
    main()
```

<br>

### Example 1

- 기본적인 형태

터미널 Input : `C:\Users\LSH\Desktop\python_basic> python test.py --x=5.5 --y=DEF --z=7`

Output :

```python
x =  5.5
y =  DEF
z =  7
```

<br>

### Example 2

- z가 required=True임에도 불구하고, 아무런 argument도 입력하지 않았을 경우

터미널 Input : `C:\Users\LSH\Desktop\python_basic> python test.py`

Output :

```
usage: test.py [-h] [--x X] [--y Y] --z Z
test.py: error: the following arguments are required: --z
```

<br>

### Example 3

- z가 required=True니까 반드시 명시적으로 입력해줘야 한다 ( 나머지는 안해도 OK )

터미널 Input : `C:\Users\LSH\Desktop\python_basic> python test.py --z=5`

Output :

```python
x =  1.0
y =  ABC
z =  5
```

<br>

### Example 4

- y는 ABC, DEF중 하나로 입력해야하지만, abc로 입력한 경우

터미널 Input : `C:\Users\LSH\Desktop\python_basic> python test.py --y=abc --z=5`

Output : 

```
usage: test.py [-h] [--x X] [--y {ABC,DEF}] --z Z
test.py: error: argument --y: invalid choice: 'abc' (choose from 'ABC', 'DEF')
```

<br>

# 3. Logging

- reference : https://www.daleseo.com/python-logging/

<br>

Logging의 우선순위 ( 덜 심각 < 더 심각 )

- debug < info < warning < error < critical

- (default level로는) warning부터가 심각한 것

  ( = warning 수준부터 메세지가 뜬다 )

<br>

## 3-1. Logging 기본

```python
import logging

def main():    
    logging.debug("A") # 출력 안됨
    logging.info("B") # 출력 안됨
    logging.warning("C") #------ 출력 됨 (default) -------
    logging.error("D")  # 출력 됨
    logging.critical("E") # 출력 됨

if __name__ == "__main__":
    main()
```

<br>

Output :

```
WARNING:root:C
ERROR:root:D
CRITICAL:root:E
```

<br>

## 3-2. Logging Level 변경

`logging.basicConfig(level=xxxx)`

```python
import logging
logging.basicConfig(level=logging.INFO)

def main():    
    logging.debug("A") # 출력 안됨
    logging.info("B") #------ 출력 됨 ----------
    logging.warning("C") # 출력 됨
    logging.error("D") # 출력 됨
    logging.critical("E") # 출력 됨

if __name__ == "__main__":
    main()
```

<br>

```
INFO:root:B
WARNING:root:C
ERROR:root:D
CRITICAL:root:E
```

<br>

## 3-3. Logging Format 변경

- `(message)`부분에 해당 값이 들어간다
- `(asctime)` : time ( `datefmt`로 조정 가능 )

```python
import logging

logging.basicConfig(level=logging.DEBUG,
                    format="'%(asctime)s - %(message)s'")

def main():    
    logging.debug("A") 
    logging.info("B") 
    logging.warning("C")
    logging.error("D") 
    logging.critical("E")

if __name__ == "__main__":
    main()
```

<br>

Output :

```
'2021-07-30 16:12:49,498 - A'
'2021-07-30 16:12:49,498 - B'
'2021-07-30 16:12:49,499 - C'
'2021-07-30 16:12:49,500 - D'
'2021-07-30 16:12:49,500 - E'
```

<br>

```python
import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

def main():    
    logging.debug("A") 
    logging.info("B") 
    logging.warning("C")
    logging.error("D") 
    logging.critical("E")

if __name__ == "__main__":
    main()
```

<br>

Output :

```
07/30/2021 16:33:53 - INFO - root -   B
07/30/2021 16:33:53 - WARNING - root -   C
07/30/2021 16:33:53 - ERROR - root -   D
07/30/2021 16:33:53 - CRITICAL - root -   E
```

<br>

## 3-4. 예외 Logging

`logging.exception("0으로 나눌수 없음")` 

```python
import logging

def calc():
    99 / 0

def main():
    try:
        calc()
    except Exception:
        logging.exception("0으로 나눌수 없음")

if __name__ == "__main__":
    main()
```

<br>

Output :

```python
ERROR:root:0으로 나눌수 없음
Traceback (most recent call last):
  File "test.py", line 10, in main
    calc()
  File "test.py", line 5, in calc
    99 / 0
ZeroDivisionError: division by zero
```

<br>

## 3-5. 유동적 Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)

def main():
    level1 = logging.DEBUG
    level2 = logging.WARNING
    level3 = logging.ERROR  
    
    logging.log(level1, "Debugging 메시지")
    logging.log(level2, "Warning 메시지")
    logging.log(level3, "Error 메시지")
    
if __name__ == "__main__":
    main()
```

<br>

Output : 

```
DEBUG:root:Debugging 메시지
WARNING:root:Warning 메시지
ERROR:root:Error 메시지
```

<br>

 # 4. OS

```python
import os
```

`os.mkdir` : 경로 생성 ( make directory )

`os.chdir` : 경로 변경 ( change directory )

`os.getcwd` : 현재 경로 불러오기 ( get current working directory )

<br>

### Example 1

- `MY_NEW_FOLDER`라는 이름의 폴더를 만든 뒤, 해댕 폴더로 경로 변경

```python
print(os.getcwd())

os.mkdir("MY_NEW_FOLDER")
os.chdir('MY_NEW_FOLDER')
print(os.getcwd())
```

Output :

```
C:\Users\LSH\Desktop\python_basic
C:\Users\LSH\Desktop\python_basic\MY_NEW_FOLDER
```

<br>

### Example 2

```python
import os

os.chdir('C:\\Users\\LSH\\Desktop\\python_basic')
print(os.getcwd())
os.chdir('../') # 상위 경로로 이동하기
print(os.getcwd())
```

Output :

```
C:\Users\LSH\Desktop\python_basic
C:\Users\LSH\Desktop
```

<br>

### Example 3

```python
data_dir='C:\\Users\\LSH\\Desktop\\python_basic'
data_name = 'MY_DATA.csv'

data1 = os.path.join(data_dir, data_name)
data2 = data_dir + '\\' + data_name

print(data1)
print(data2)
```

Output :

```
C:\Users\LSH\Desktop\python_basic\MY_DATA.csv
C:\Users\LSH\Desktop\python_basic\MY_DATA.csv
```

<br>

# 5. tqdm

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

<br>

# 6. csv & pickle

## 6-1.csv

```python
import csv
```

- `pd.read_csv`외에, csv를 불러오는데에 가끔가다 사용되는 패키지

<br>

Example file : `HAN.csv`

내용 :

```
Hello
My Name Is Seung Han Lee
Nice To Meet You
```

<br>

### How to read csv files using `csv` packages?

```python
import csv

# (1) delimiter = ' ' 
with open('HAN.csv',newline='') as file:
    data = csv.reader(file,delimiter=' ')
    for line in data:
        print(line)

print('------------------------ \n ') 

# (2) delimiter = '\t' ( = tab)
with open('HAN.csv',newline='') as file:
    data = csv.reader(file,delimiter='\t')
    for line in data:
        print(line)

print('------------------------ \n ') 

# (3) delimiter = '\t' ( = tab) + join
with open('HAN.csv',newline='') as file:
    data = csv.reader(file,delimiter='\t')
    for line in data:
        print(''.join(line))
```

Output :

```
['Hello']
['My', 'Name', 'Is', 'Seung', 'Han', 'Lee']
['Nice', 'To', 'Meet', 'You']
------------------------

['Hello']
['My Name Is Seung Han Lee']
['Nice To Meet You']
------------------------

Hello
My Name Is Seung Han Lee
Nice To Meet You
```

<br>

## 6-2. pickle

```python
import pickle
```

- 피클(pkl) 파일을 read/write하는데에 사용

<br>

### (1) Write

```python
temp = ['a', 'b', 'c']

with open('newfiles2.pkl', 'wb') as file:
    pickle.dump(temp, file)
```

<br>

### (2) Read

```python
with open('newfiles2.pkl', 'rb') as file:
    data = pickle.load(file)
    print(data)
```

<br>

Output :

```
['a', 'b', 'c']
```



