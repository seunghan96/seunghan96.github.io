# [ OS  ]

```python
import os
```

`os.mkdir` : 경로 생성 ( make directory )

`os.chdir` : 경로 변경 ( change directory )

`os.getcwd` : 현재 경로 불러오기 ( get current working directory )

<br>

# Example 1

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

# Example 2

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

# Example 3

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

