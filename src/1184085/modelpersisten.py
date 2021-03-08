# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 21:36:58 2021

@author: USER
"""

from sklearn import svm #mengimport svm dari scikit learn library
from sklearn import datasets #mengimport dataset dari scikit learn library
clf = svm.SVC(gamma=0.001, C=100.) #memanggil class SVC dan manset argument sonstructor SVC serta ditampung di variable clf
X, y= datasets.load_iris(return_X_y=True) 
#meload dataset iris dan datasets dan ditampung di variable x untuk data dan y untuk target

clf.fit(X, y) #memanggil methode fit untuk melakukan training data dengan argumen data dan target dari datasets iris

#pickle
import pickle #mengimport pickle
s = pickle.dumps(clf) #memanggil method dumps dengan argumen clf dan ditampung di variable s
clf2 = pickle.loads(s) #memanggil metnode loads dengan argumen s dan ditampung di variable clf2
print(clf2.predict(X[0:1])) #menampilkan hasil dari method predict dengan argumen data variable

from joblib import dump, load #mengimport dump dan load dari library joblib
dump(clf, '1184085.joblib')  #memanggil methode dumps dengan argumen clf dan nama file joblibnya
clf3 = load('1184085.joblib') #memanggil methode loads dengan argumen nama file joblinya dan ditampung di variable clf3
print(clf3.predict(X[0:1])) #menampilkan hasil dari methode predict dengan argumen data variable x pertama