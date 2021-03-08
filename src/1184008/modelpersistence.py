# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 21:31:52 2021

@author: GANY
"""
from sklearn import svm
from sklearn import datasets
clf = svm.SVC(gamma=0.001, C=100.)
X, y= datasets.load_iris(return_X_y=True)
clf.fit(X, y)

import pickle #import pickle
s = pickle.dumps(clf) #memanggil method dumps dengan argumen clf dan ditampung di variable s
clf2 = pickle.loads(s) #memanggil method loads dengan argumen s ditampung di variable clf2
print(clf2.predict(X[0:1])) #menampilkan hasil dan method predict dengan argumen data variable X pertama

from joblib import dump, load #import dump dan load dari si hoblib
dump(clf, '1184008.joblib') #memanggil method dumps dengan argumen clf dan nama file joblibnya
clf3 = load('1184008.joblib') #memanggil method loads dengan argumen nama file joblibnya dan ditampung divariable clf3
print(clf3.predict(X[0:1])) #menampilkan hasil dari method predict dengan argumen data variable X pertama