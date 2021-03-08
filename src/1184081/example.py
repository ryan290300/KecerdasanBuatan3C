# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:04:18 2021

@author: ASUS
"""

print(__doc__)

from sklearn.linear_model import LogisticRegression
from sklearn import set_config


lr = LogisticRegression(penalty='11')
print('Default representation:')
print(lr)

set_config(print_changed_only=True)
print('\nWith changed_only option:')
print(lr)