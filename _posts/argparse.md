# [ Argparse ]

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

# Example 1

- 기본적인 형태

터미널 Input : `C:\Users\LSH\Desktop\python_basic> python test.py --x=5.5 --y=DEF --z=7`

Output :

```python
x =  5.5
y =  DEF
z =  7
```

<br>

# Example 2

- z가 required=True임에도 불구하고, 아무런 argument도 입력하지 않았을 경우

터미널 Input : `C:\Users\LSH\Desktop\python_basic> python test.py`

Output :

```
usage: test.py [-h] [--x X] [--y Y] --z Z
test.py: error: the following arguments are required: --z
```

<br>

# Example 3

- z가 required=True니까 반드시 명시적으로 입력해줘야 한다 ( 나머지는 안해도 OK )

터미널 Input : `C:\Users\LSH\Desktop\python_basic> python test.py --z=5`

Output :

```python
x =  1.0
y =  ABC
z =  5
```

<br>

# Example 4

- y는 ABC, DEF중 하나로 입력해야하지만, abc로 입력한 경우

터미널 Input : `C:\Users\LSH\Desktop\python_basic> python test.py --y=abc --z=5`

Output : 

```
usage: test.py [-h] [--x X] [--y {ABC,DEF}] --z Z
test.py: error: argument --y: invalid choice: 'abc' (choose from 'ABC', 'DEF')
```

