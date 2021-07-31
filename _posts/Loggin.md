# [ LOGGING ]

- reference : https://www.daleseo.com/python-logging/

<br>

Logging의 우선순위 ( 덜 심각 < 더 심각 )

- debug < info < warning < error < critical

- (default level로는) warning부터가 심각한 것

  ( = warning 수준부터 메세지가 뜬다 )

<br>

# 1. Logging 기본

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

# 2. Logging Level 변경

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

# 3. Logging Format 변경

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

# 4. 예외 Logging

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

# 5. 유동적 Logging

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

