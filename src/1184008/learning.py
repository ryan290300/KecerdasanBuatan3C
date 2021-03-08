# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 21:17:50 2021

@author: GANY
"""
from sklearn import svm #perintah untuk mengimport class svm dari package sklearn
from sklearn import datasets #perintah untuk mengimport dataset
digits = datasets.load_digits() #memasukan dataset digits ke variable digits

clf = svm.SVC(gamma=0.001, C=100.) #cll as parameter, svm.SVC sebagai class, dan untuk parameter untuk menetapkan nilai secara manual adalah gamma

clf.fit(digits.data[:-1], digits.target[:-1]) #clf sebagai parameter, fit as metode, dan si digit.data sebagai item , [:1] sebagai syntax pythonnya dan menampilkan outputnya

print(clf.predict(digits.data[-1:])) #clf sebagai parameter, predict untuk metode lainnnya, digits . data sebagai item outputnnya