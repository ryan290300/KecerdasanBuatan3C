# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:18:27 2021

@author: ASUS
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
dump(clf, '1184081 joblib')
clf3 = load('1184081 joblib')
print(clf3.predict(X[0:1]))