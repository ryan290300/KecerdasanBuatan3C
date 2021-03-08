# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 21:24:16 2021

@author: USER
"""

from sklearn import svm #perintah untuk mengimport class svm dari package sklearn
digits = datasets.load_digits() #memuat dan memasukkan dataset digits ke variable digits
clf = svm.SVC(gamma=0.001, C=100.) 
#clf sbg estimator/parameter. svm.SVC sbg class. gamma sbg parameter untuk menetapkan nilai secara manual

clf.fit(digits.data[:-1], digits.target[:-1]) 
#clf sbg estimator/parameter. fit sbg metode. digits.data sbg item. [:1] sbg syntax python dan menampilkan outputnya.

print(clf.predict(digits.data[-1:])) 
#clf sbg estimator/parameter. predict sbg metode  lainnya. digits.data sbg item dan menampilkan outputnya.