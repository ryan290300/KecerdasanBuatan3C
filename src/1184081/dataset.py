# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:11:20 2021

@author: ASUS
"""

from sklearn import datasets()
iris = datasets.load_iris()
digits = datasets.load_digits()

print(digits.data)

digits.target

digits.image[0]