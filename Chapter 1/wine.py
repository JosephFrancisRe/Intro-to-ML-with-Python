# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 08:31:37 2022

@author: JosephRe
"""

import sklearn
from sklearn.datasets import load_wine

wine_dataset = load_wine()

print("keys: ", wine_dataset.keys(), "\n")
print("data: ", wine_dataset['data'], "\n")
print("target: ", wine_dataset['target'], "\n")
print("frame: ", wine_dataset['frame'], "\n")
print("target_names: ", wine_dataset['target_names'], "\n")
print("DESCR: ", wine_dataset['DESCR'], "\n")
print("feature_names: ", wine_dataset['feature_names'], "\n")

print(wine_dataset['data'].shape, "\n")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(wine_dataset['data'], wine_dataset['target'], random_state=0)

print("X_train size = ", X_train.size, "| X_test size = ", X_test.size, ".\n")

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

import numpy as np

accuracy = np.mean(y_pred == y_test)

print("knn accuracy v1: ", accuracy, "\n")
print("knn accuracy v2: ", knn.score(X_test, y_test), "\n")

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
print("lr accuracy: ", lr.score(X_test, y_test), "\n")

from sklearn import svm

svm_object = svm.SVC(decision_function_shape='ovo')
svm_object.fit(X_train, y_train)
print("svm accuracy: ", svm_object.score(X_test, y_test), "\n")

from sklearn import tree

tree_object = tree.DecisionTreeClassifier()
tree_object.fit(X_train, y_train)
print("decision tree accuracy: ", tree_object.score(X_test, y_test))