# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:50:11 2021

@author: Aditya Luthfi
"""

from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()

print(digits.data)

digits.target

digits.images[0]