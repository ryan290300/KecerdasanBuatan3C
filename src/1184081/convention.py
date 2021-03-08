# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:23:19 2021

@author: ASUS
"""

import numpy as np
from sklearn import random_projection
rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
X = np.array(X, dtype='float32')
print(X.dtype)
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new.dtype)

from sklearn import datasets
from sklearn.svm import SVC
iris = datasets.load_iris()
clf = svm.SVC(gamma=0.001, C=100.)