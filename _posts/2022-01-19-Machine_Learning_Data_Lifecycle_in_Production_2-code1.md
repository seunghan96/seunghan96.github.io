---
title: \[Week 2-code 1\] Simple Feature Engineering
categories: [MLOPS]
tags: []
excerpt: (coursera) Machine Learning Data Lifecycle in Production - Feature Engineering, Transformation and Selection
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

( reference : [Machine Learning Data Lifecycle in Production](https://www.coursera.org/learn/machine-learning-data-lifecycle-in-production) ) 

# Simple Feature Engineering

Goal : ***Tensorflow Transform***에 익숙해지기

### Contents

1. Collect **Raw Data**
2. Define **Meta Data**

3. **Preprocessing function**
4. **Constant graph**, with required transformation

<br>

# 1. Collect **Raw Data**

```python
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam

from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils

import pprint
import tempfile
```

<br>

다음과 같이 3개의 변수를 가진 3개 행의 데이터를 정의한다.

```python
raw_data = [
      {'x': 1, 'y': 1, 's': 'hello'},
      {'x': 2, 'y': 2, 's': 'world'},
      {'x': 3, 'y': 3, 's': 'hello'}
  ]
```



# 2. Define **Meta Data**

Meta Data : 각 변수에 대한 schema 정보를 담고 있다.

세부 내용

- (a) 생성될 meta data는, `DatasetMetadata` 객체에 pack 될 것

  ( 그런 뒤, transform data를 거치게 될 것 )

- (b) `DatasetMetadata` 객체는 “Schema protocl buffer” 데이터 형식을 입력으로 받는다

  - `schema_from_feature_spec()` 을 사용 ( 인풋 : 딕셔너리 형식 )

- (c) 위의 딕셔너리를 만들 때, `FeatureSpecType` 를 value로써 할당해야!

  - 데이터 형식을 지정해줌

<br>

```python
raw_data_metadata = dataset_metadata.DatasetMetadata( #-------------- (a)
  schema_utils.schema_from_feature_spec({ #-------------------------- (b)
    'y': tf.io.FixedLenFeature([], tf.float32), #-------------------- (c)
        'x': tf.io.FixedLenFeature([], tf.float32),
        's': tf.io.FixedLenFeature([], tf.string),
  })
)
```

<br>

이렇게 생성된 **Meta Data**의 schema를 확인해보자.

```python
print(raw_data_metadata._schema)
```

![figure2](/assets/img/mlops/img130.png)

<br>

# 3. **Preprocessing function**

**전처리 함수 (preprocessing function)** 은 `tf.Transform` 의 핵심이다.

- input & output : ***dictionary of tensors***
  - 여기서의 tensor는 `Tensor` 혹은 `SparseTensor` 형식이다.

<br>

2 main groups of API calls

- (1) **TensorFlow Ops** : training & serving 둘 다

  - one feature vector at a time

- (2) **TensorFlow Transform Analyzers** : training 때만

  - WHOLE traning data

  - create **tensor constants**

    ( inference 단에서, 이 constant는 “불변한다” )

<br>

```python
def preprocessing_fn(inputs):
    x = inputs['x']
    y = inputs['y']
    s = inputs['s']
    
    # TRANSFORMATION
    x_centered = x - tft.mean(x)
    y_normalized = tft.scale_to_0_1(y)
    s_integerized = tft.compute_and_apply_vocabulary(s)
    x_centered_times_y_normalized = (x_centered * y_normalized)
    
    return {
        'x_centered': x_centered,
        'y_normalized': y_normalized,
        's_integerized': s_integerized,
        'x_centered_times_y_normalized': x_centered_times_y_normalized,
    }
```

<br>

# 4. **Constant graph**, with required transformation

배포의 확장성 & 유연성을 위해, **Apache Beam** 사용 가능

- pipe (|) operator를 사용하여 연쇄적으로 적용

```python
tf.get_logger().setLevel('ERROR')

with tft_beam.Context(temp_dir=tempfile.mkdtemp()):    
    # pipeline 정의하기
    transformed_dataset, transform_fn = (
        (raw_data, raw_data_metadata) | tft_beam.AnalyzeAndTransformDataset(
            preprocessing_fn) )

transformed_data, transformed_metadata = transformed_dataset
```

<br>

Transform 된 결과 확인

```python
print('\nRaw data:\n{}\n'.format(pprint.pformat(raw_data)))
print('Transformed data:\n{}'.format(pprint.pformat(transformed_data)))
```

![figure2](/assets/img/mlops/img131.png)

