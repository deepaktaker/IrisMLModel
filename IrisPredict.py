#!/usr/bin/env python
# coding: utf-8

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn2pmml import PMMLPipeline

iris = load_iris()
X_train, X_test, y_train, y_test =  train_test_split(iris.data, iris.target, test_size = 0.25)
y_train
pipe_l = PMMLPipeline([('minmax', MinMaxScaler()), ('lr', LogisticRegression())])
model = pipe_l.fit(X_train,y_train)
pipe_l.score(X_test, y_test)
X_test[1]
y_test[1]
pipe_l.predict([[6.7, 3. , 5.2, 2.3]])

import pandas as pd
irisd = pd.DataFrame(iris.data, columns=iris.feature_names)
irisd.columns

from nyoka import skl_to_pmml
skl_to_pmml(pipe_l, irisd.columns, 'Target', "iris_nyoka_pipeline.pmml")

from sklearn2pmml import sklearn2pmml
sklearn2pmml(pipe_l, "iris_pipeline.pmml", with_repr = True, debug = True)

