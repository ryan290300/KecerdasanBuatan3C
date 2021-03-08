# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 19:09:30 2021

@author: Muh Amri
"""

from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
print(digits.data)

from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1])
clf.predict(digits.data[:-1])