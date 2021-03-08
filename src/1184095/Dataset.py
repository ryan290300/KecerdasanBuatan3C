# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 21:16:18 2021

@author: Dian
"""

from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()

print(digits.data)

digits.target

digits.images[0]