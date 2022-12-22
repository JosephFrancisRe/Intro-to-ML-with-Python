# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 02:19:39 2022

@author: JosephRe
"""

import mglearn

X, y = mglearn.datasets.make_forge()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

print("The accuracy of the model is", knn.score(X_test, y_test), "\n")

mglearn.plots.plot_knn_classification(n_neighbors = 3)

import matplotlib.pyplot as plt

plt.legend(loc = 4)
plt.xlabel('First feature')
plt.ylabel('Second feature')
plt.title('Plotting kNN Model')