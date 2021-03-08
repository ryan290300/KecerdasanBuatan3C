# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:15:05 2021

@author: ASUS
"""

from sklearn import svm
digits = datasets.load_digits()
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1])
print(clf.predict(digits.data[-1:]))