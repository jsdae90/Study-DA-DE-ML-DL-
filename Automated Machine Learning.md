# Automated Machine Learning



|                              | Bayesian<br> Optimization                                                                                                                                                                              | Reinforcement<br> Learning                                                              | Evolutionary<br> Algorithm                                                                                      | Gradient<br> based                                                                                                | Framework                                                                                                         |
|:----------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|
| 증강<br>특성<br>공학<br>           |                                                                                                                                                                                                        | FeatureRL                                                                               | 특성 공학용 GP<br>(유전프로그래밍))                                                                                         |                                                                                                                   | FeatureTools                                                                                                      |
| 자동 모델과 <br>하이퍼파라미터<br>검색     | TPE<br>(트리 기반 파젠 추정기)<br> <br>SMAC<br>(일반적 알고리즘 설정을 위한 순차적 모델 기반 최적화)<br> <br>Auto-SKLearn<br> <br>FABOLAS<br>(대규모 데이터셋에서 머신러닝 하이퍼파라미터의 빠른 베이지안 최적화)<br> <br>BOHB<br>(규모 확장 가능한 강건하고 효율적인 하이퍼파라미터 최적화) | APRL<br>(강화학습을 통한 자율 예측적 모델)<br> <br>Hyperband<br>(하이퍼파라미터 최적화에 대한 혁신적인 밴딧 기반 접근법)      | TPOT<br>(기반 파이프라인 최적화<br> <br>AutoStacker<br>(자동 진화 계층적 머신러닝 시스템)<br> <br>DarwinML<br>(자동머신러닝용 그래프 기반 진화 알고리즘)) |                                                                                                                   | Hyperopt<br>(분산 비동기적 하이퍼파라미터 최적화)<br> <br>SMAC<br>(일반적 알고리즘 설정을 위한 순차적 모델 기반 최적화)<br> <br> TPOT<br>(기반 파이프라인 최적화) |
| 자동 딥러닝<br>또는<br>신경망<br>구조 탐색 | AutoKeras<br> <br>NASBot                                                                                                                                                                               | NAS<br>(신경망 구조 탐색)<br> <br>NASNET(신경망 구조 탐색 네트워크)<br> <br>ENAS<br>(파라미터 공유를 통한 효율적 NAS) |                                                                                                                 | DARTS<br>(미분 가능 구조 탐색)<br> <br>ProxylessNAS<br>(타깃 작업과 하드웨어에 대한 직접 신경망 구조 탐색)<br> <br>NAONet<br>(신경망 구조 최적화 네트워크) | AutoKeras<br> <br>AdaNet<br> <br>NNI<br>(신경망 지능)                                                                  |

<center>**<AutoML 기법>**</center>

<br>

<br>

<br>

<br>

## 오픈소스 툴과 라이브러리를 활용한 AutoML

|                  | 언어     | AutoML기법         | 자동 특성 추출 | 메타러닝 | 링크                                              |
| ---------------- | ------ | ---------------- | -------- | ---- | ----------------------------------------------- |
| AutoWeka         | Java   | 베이지안 최적화         | 예        | 아니요  | https://github.com/automl/autoweka              |
| AutoSklearn      | Python | 베이지안 최적화         | 예        | 예    | https://automl.github.io/auto-sklearn/master/   |
| TPOT             | Python | 유전 알고리즘          | 예        | 아니요  | https://epistasislab.github.io/tpot             |
| Hyperopt-Sklearn | Python | 베이지안 최적화와 랜덤 탐색  | 예        | 아니요  | https://github.com/hyperopt/hyperopt-sklearn    |
| AutoStacker      | Python | 유전 알고리즘          | 예        | 아니요  | https://arxiv.org/abs/1803.00684                |
| AlphaD3M         | Python | 강화학습             | 예        | 예    | https://www.cs.columbia.edu/~idori/AlphaD3M.pdf |
| OBOE             | Python | 유전 알고리즘          | 아니요      | 예    | https://github.com/udellgroup/oboe              |
| PMF              | Python | 협업 필터링과 베이지안 최적화 | 예        | 예    | https://github.com/rsheth80/pmf-automl          |

<center><AutoML Framework의 특징들></center>

<br>

<br>

<br>

<br>

### AutoML TPOT Example(mnist)

```python
!pip install TPOT

from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                                          train_size = 0.75, test_size = 0.25)
X_train.shape, X_test.shape, y_train.shape

# 1 minute test
# verbosity = 2,  TPOT Classifier Parameter Set 표기
tpot = TPOTClassifier(verbosity = 2, max_time_min = 1, population_size = 40)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))


# 5 minutes test
tpot = TPOTClassifier(verbosity = 2, max_time_min = 5, population_size = 40)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

# 25 minutes test
tpot = TPOTClassifier(verbosity = 2, max_time_min = 25, population_size = 40)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

# 1 hour test
tpot = TPOTClassifier(verbosity = 2, max_time_min = 60, population_size = 40)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

# Model Export
tpot.export('tpot_digits_pipeline.py')
```

<center><Mnist Model Export Using Tpot></center>

<br>

<br>

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, VariancThresholdfrom 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

# NOTE : Make sure that the outcome columns is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis = 1)
training_features, testing_features, training_target, testing_target =
                   train_test_split(features, tpot_data['target'], random_state = None)

# Average CV score on the training set was : 0.990356601955115
exported_pipeline = make_pipeline(
                         RFE(estimator=ExtraTreesClassifier(criterion = "gini",
                                                            max_features = 0.700000000000001,
                                                            n_estimators = 100),                                                            step = 0.2),
                         VarianceThreshold(threshold = 0.0001),
                         KNeighborsClassifier(n_neighbors = 2, p = 2, weights = "distance")
)


exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
```

<center><AutoML TPOT Pipeline></center>

<br>

<br>

## AutoML FeatureTools DFS Example(Boston House Price Predict)

```python
!pip install featuretools


from sklearn.datasets import load_boston
import pandas as pd
import featuretools as ft

# load data and put into dataframe
boston = load_boston()
df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['MEDV'] = boston.targetprint (df.head(5))

# Make an entityset and add the entity
es = ft.EntitySet(id = 'boston')
es.entity_from_dataframe(entity_id = 'data', dataframe = df,
                         make_index = True, index = 'index')


# Run deep feature synthesis with transformation primitives
# FeatureTools가 dfs실행을 통해 모든 합과 곱의 특성을 갖게 됨.
# 이를 통해, 여러 데이터 포인트간의 잠재적인 관계를 강조하는 것으로 생각할 수 있음.
feature_matrix, feature_defs = ft.dfs(entityset = es, target_entity = 'data',
                                      trans_primitivees = ['add_numeric', 'multiply_numeric'])

feature_matrix.head()
```

<br>

<br>

## AutoML auto-sklearn

```python
# auto-sklearn 기본 코드
# auto-sklearn은 자동 앙상블 구축을 통해 베이지안 최적화를 지원
import autosklearn.classification
cls = autosklearn.classification.AutoSklearnClassifier()
cls.fit(X_train, y_train)
predictions = cls.predict(X_test)
```

<center><auto-sklearn 기본 코드></center>

<br>

<br>

```python
!apt-get install swig -y
!pip install Cython numpy
!pip install auto-sklearn
!pip install liac-arff

import autosklearn.classification
import sklearn.model_selection as cv
import sklearn.datasets
import sklearn.metrics
#from autosklearn.experimental.askl2 import AutoSklearn2Classifier

x, y = sklearn.datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = \
         sklearn.model_selection.train_test_split(X, y, random_state = 1)
automl = autosklearn.classification.AutoSklearnClassifier()
automl.fit(X_train, y_train)

```

<center><auto-sklearn을 이용한 AutoML - 분류기 간단 실험></center>


