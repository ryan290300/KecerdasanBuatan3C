# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:11:06 2021

@author: Putri Nella
"""
from sklearn import datasets #import class dataset dari si scikit learn librari
iris = datasets.load_iris() #memasukan dataset iris ke variable bernama iris
digits = datasets.load_digits() #memasukan dataset digit ke variable digits

print(digits.data) #memberikan akses ke fitur yang dapat digunakan untuk mengklasifikasikan sample digit dan agar tertampil di console

digits.target #memberikan informasi tentang data yang berhubungan atau dijadikan sebagai label

digits.images[0] #data berupa array 2D, shape ( n.samles, n.features), meskipun data asliya, mungkin memiliki bentuk yang berbeda