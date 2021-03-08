# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 19:45:40 2021

@author: user
"""

import numpy as np
from sklearn import random_projection

rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
X = np.array(X, dtype='float32')
X.dtype

transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
X_new.dtype
print(X_new)