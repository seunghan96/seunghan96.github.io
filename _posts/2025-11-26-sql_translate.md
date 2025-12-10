---
title: (OpenAI API) SQL Translate
categories: [LLM]
tags: []
excerpt: 

---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# (OpenAI API) SQL Translate

**OpenAI의 SQL Translate 기능**

<br>

# **1. SQL Translate란?**

**SQL Translate =** 자연어 입력을 SQL 쿼리로 정확하게 변환하기 위해 LLM을 사용

- 사람: “자연어로 질문”을 하면
- 모델: **SQL 쿼리 형태로 변환해주는 역할**

<br>

Example) 

“Show me all users who signed up last week.”

```sql
SELECT *
FROM users
WHERE signup_date >= DATE('now', '-7 days');
```

이 기능은 OpenAI API의 특정 모드가 아니라,

**모델을 SQL 생성에 맞춰 prompt/responses.create()로 사용하는 패턴**

<br>

# **2. SQL Translate의 실제 용도**

- 데이터 분석 요청을 SQL로 자동 변환
- 대시보드/BI툴에서 자연어 질의를 SQL로 바꿔주는 기능
- 백엔드 서비스에서 자연어 기반 DB query 인터페이스 제공
- SQL 초보자 사용자도 자연어로 데이터 질의 가능

<br>

# **3. 공식 문서 Python 예시**

https://platform.openai.com/docs/examples/default-sql-translate

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
  model="gpt-4o",
  input=[
    {
      "role": "system",
      "content": [
        {
          "type": "input_text",
          "text": "Given the following SQL tables, your job is to write queries given a user’s request.\n    \n    CREATE TABLE Orders (\n      OrderID int,\n      CustomerID int,\n      OrderDate datetime,\n      OrderTime varchar(8),\n      PRIMARY KEY (OrderID)\n    );\n    \n    CREATE TABLE OrderDetails (\n      OrderDetailID int,\n      OrderID int,\n      ProductID int,\n      Quantity int,\n      PRIMARY KEY (OrderDetailID)\n    );\n    \n    CREATE TABLE Products (\n      ProductID int,\n      ProductName varchar(50),\n      Category varchar(50),\n      UnitPrice decimal(10, 2),\n      Stock int,\n      PRIMARY KEY (ProductID)\n    );\n    \n    CREATE TABLE Customers (\n      CustomerID int,\n      FirstName varchar(50),\n      LastName varchar(50),\n      Email varchar(100),\n      Phone varchar(20),\n      PRIMARY KEY (CustomerID)\n    );"
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "Write a SQL query which computes the average total order value for all orders on 2023-04-01."
        }
      ]
    }
  ],
  temperature=1,
  max_output_tokens=1024
)
```

<br>

**모델 출력 예시:**

````
To compute the average total order value for all orders on 2023-04-01, we need to join the Orders, OrderDetails, and Products tables. We will sum the product of Quantity and UnitPrice for each order, and then compute the average of these sums.

Here is the SQL query:

```sql
SELECT AVG(TotalOrderValue) AS AverageOrderValue
FROM (
  SELECT Orders.OrderID, SUM(OrderDetails.Quantity * Products.UnitPrice) AS TotalOrderValue
  FROM Orders
  JOIN OrderDetails ON Orders.OrderID = OrderDetails.OrderID
  JOIN Products ON OrderDetails.ProductID = Products.ProductID
  WHERE Orders.OrderDate = '2023-04-01'
  GROUP BY Orders.OrderID
) AS OrderValues;
```

This query first computes the total value of each order in the subquery, and then computes the average of these values in the outer 
````