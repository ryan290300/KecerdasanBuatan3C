# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 22:15:39 2021

@author: Putri Nella
"""

from sklearn import svm
from sklearn import datasets
clf = svm.SVC(gamma=0.001, C=100.)
X, y= datasets.load_iris(return_X_y=True)
clf.fit(X, y)

import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
print(clf2.predict(X[0:1]))

from joblib import dump, load
dump(clf, '1184017.joblib') 
clf3 = load('1184017.joblib')
print(clf3.predict(X[0:0]))