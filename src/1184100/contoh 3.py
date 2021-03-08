# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 19:43:41 2021

@author: Muh Amri
"""

#%% cara pertama
from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
X, y = datasets.load_iris(return_X_y=True)
clf.fit(X, y)

import pickle 
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0:1])
print(y[0])

#%% cara kedua
from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
X, y = datasets.load_iris(return_X_y=True)
clf.fit(X, y)

from joblib import dump, load
dump(clf, '1184100.joblib')
clf3 = load('1184100.joblib')
print(clf3.predict(X[0:1]))



