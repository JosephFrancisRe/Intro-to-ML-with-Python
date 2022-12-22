# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sklearn

from sklearn.datasets import load_iris

iris_dataset = load_iris()
print("keys of the iris_dataset:\n", iris_dataset.keys(), "\n\n")

print("description of the iris_dataset:\n", iris_dataset["DESCR"], "\n\n")

print("target names: ", iris_dataset["target_names"], "\n\ntarget values: ", iris_dataset["target"], "\n\n")

print("dimensions of the iris_dataset: ", iris_dataset['data'].shape)

print("first five feature values:\n", iris_dataset['feature_names'], "\n", iris_dataset['data'][:5], "\n\n")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print("dimensions of training set's data': ", X_train.shape, "\n")
print("dimensions of testing set's data': ", X_test.shape, "\n")
print("dimensions of training set's target': ", y_train.shape, "\n")
print("dimension of testing set's target': ", y_test.shape, "\n")

import pandas as pd
import mglearn

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(X_train, y_train)

import numpy as np
    
X_new = np.array([[5, 2.9, 1, 0.2]])

prediction = knn.predict(X_new)

print("\n\nPrediction: ", prediction, ", which is called ", iris_dataset['target_names'][prediction], "\n\n")

y_pred = knn.predict(X_test)

print("testing set's predictions: ", y_pred, "\n\n")

accuracy = np.mean(y_test == y_pred)

print("Results of predictions (v1): {:.2f}".format(accuracy), "\n\n")

accuracy2 = knn.score(X_test, y_test)

print("Results of predictions (v2): {:.2f}".format(accuracy2))