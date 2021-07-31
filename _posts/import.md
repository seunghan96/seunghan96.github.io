# [ Import Custom Packages ]

<br>

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

# Example 1

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

# Example 2

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

# Example 3

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

