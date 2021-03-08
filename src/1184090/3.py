# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 20:14:54 2021

@author: Aditya Luthfi
"""

from sklearn import svm
from sklearn import datasets
clf = svm.SVC(gamma=0.001, C=100.)
X, y= datasets.load_iris(return_X_y=True)
clf.fit(X, y)

import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0:1])
prin(clf2.predict(X[0:1]))

from joblib import dump, load
dump(clf, '1184090,joblib')
clf3 = load('1184090,joblib')
print(clf3.predict(X[0:1]))