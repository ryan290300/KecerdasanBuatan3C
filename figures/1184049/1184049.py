# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 14:35:22 2021

@author: ASUS
"""

from sklearn import datasets #untuk memanggil class datasets
iris = datasets.load_iris() #menggunakan contoh datasets iris
digits = datasets.load_digits() #menyimpan nilai datasets iris
print(digits.data) #menampilkan hasil variabel digits


from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
print(digits.target)


from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
print(digits.images[0])



from sklearn import svm #untuk mengimport class SVM dari Library sklearn
clf = svm.SVC(gamma=0.001, C=100.)



from sklearn import svm, datasets #untuk mengimport class SVM dari data sets

digits = datasets.load_digits()
clf = svm.SVC(gamma=0.001, C=100.) #memasukan implementasi dari "Support Vector Classification" ke variabel clf
x = clf.fit(digits.data[:-1], digits.target[:-1]) #untuk melakukan pengiriman data training set ke method fit
y = clf.predict(digits.data[:-1]) #untuk melakukan prediksi nilai pada digits
print(x)
print(y)


import numpy as np #memanggil library numpy dan dibuat alias np
from sklearn import random_projection #memanggil class random_projection

rng = np.random.RandomState(0) #membuat variabel rng, mendefinisikan np, fungsi random dan attr Random
X = rng.rand(10, 2000) #untuk membuat variabel X dan menentukan nilai X
X = np.array(X, dtype='float32') #untuk menyimpan nilai random dengan tipe data float 32
X.dtype #mengubah tipe data menjadi float 64

transformer = random_projection.GaussianRandomProjection() #membuat variabel transformer dan mendefinisikan random_projcetion
X_new = transformer.fit_transform(X) #membuat variabel X yang baru
X_new.dtype #mengubah tipe data menjadi 64
print(X_new) #menampilkan X_new


from sklearn import svm, datasets

digits = datasets.load_digits() #untuk memuat dan memasukan dataset digit ke variabel digits
clf = svm.SVC(gamma=0.001, C=100.)
print(clf)


from sklearn import datasets
from sklearn.svm import SVC

iris = datasets.load_iris() #untuk memuat dan memasukan datasets iris ke variabel bernama iris
clf = SVC()
X = clf.fit(iris.data, iris.target)
print(X)


from sklearn import datasets
from sklearn.svm import SVC

iris = datasets.load_iris()
clf = SVC()
X = clf.fit(iris.data, iris.target)
print(X)

Y = list(clf.predict(iris.data[:3]))
print(Y)

Z = clf.fit(iris.data, iris.target_names[iris.target])
print(Z)

M = list(clf.predict(iris.data[:3]))
print(M)


import numpy as np 
from sklearn.datasets import load_iris
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True) #mengambil datasets iris

clf = SVC()
A = clf.set_params(kernel='linear').fit(X, y)
print(A)

B = clf.predict(X[:5])
print(B)

C = clf.set_params(kernel='rbf').fit(X, y)
print(C)

D = clf.predict(X[:5])
print(D)


from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

X = [[1, 2], [2,4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]

classif = OneVsRestClassifier(estimator=SVC(random_state=0))
pred = classif.fit(X, y).predict(X)
print(pred)


from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

X = [[1, 2], [2,4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]

y = LabelBinarizer().fit_transform(y)
classif = OneVsRestClassifier(estimator=SVC(random_state=0))
pred = classif.fit(X, y).predict(X)
print(pred)


from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

X = [[1, 2], [2,4], [4, 5], [3, 2], [3, 1]]
y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]

y = MultiLabelBinarizer().fit_transform(y)
classif = OneVsRestClassifier(estimator=SVC(random_state=0))
pred = classif.fit(X, y).predict(X)
print(pred)