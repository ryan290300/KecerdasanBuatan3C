# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 21:20:48 2021

@author: ASUS
"""

from sklearn import svm
digits = datasets.load_digits()
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1])
print(clf.predict(digits.data[-1:]))