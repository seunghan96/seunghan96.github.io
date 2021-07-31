# [ CSV & Pickle ]

# 1. CSV

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

## How to read csv files using `csv` packages?

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

# 2. Pickle	

```python
import pickle
```

- 피클(pkl) 파일을 read/write하는데에 사용

<br>

## (1) Write

```python
temp = ['a', 'b', 'c']

with open('newfiles2.pkl', 'wb') as file:
    pickle.dump(temp, file)
```

<br>

## (2) Read

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



