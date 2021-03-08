# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 21:13:55 2021

@author: USER
"""

from sklearn import datasets #mengimport class dataset dari scikit learn library
iris = datasets.load_iris() #memuat dan memasukkan dataset iris ke variable bernama iris
digits = datasets.load_digits() #memuat dan memasukkan dataset digits ke variable digits

print(digits.data) #memberikan akses ke fitur yg digunakan untuk mengklasifikasikan sampel digits

digits.target #info data berhungan atau label

digits.images[0] #Data berupa array 2D, shape(n.samples. n.features), meskipundata asli bisa saja bentuknya beda