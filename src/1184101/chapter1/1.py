#%%
#Loading an example dataset
from sklearn import datasets #digunakan untuk memanggil class datasets dari library sklearn
iris = datasets.load_iris() #menggunakan datasets iris
digits = datasets.load_digits() #menyimpan data sets iris pada variabel digits

print(digits.data) #menampilkan hasil dari variabel data
print(digits.target) #menampilkan hasil dari variabel target
print(digits.images[0]) #menampilkan hasil dari variabel images
#%%
#Learning and predicting
from sklearn import svm #untuk mengambil svm di library sklearn
clf = svm.SVC(gamma=0.001, C=100) #memberikan nilai gamma secara manual

clf.fit(digits.data[:-1], digits.target[:-1]) #csf sebagai classifier dan kemudian set latihan dengan metode fit
clf.predict(digits.data[-1:]) #memprediksi nilai baru dari digit data

#%%
#type casting
import numpy as np
from sklearn import random_projection

rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
X = np.array(X, dtype='float32')
X.dtype

transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
X_new.dtype

#regression targets

from sklearn import datasets
from sklearn.svm import SVC
iris = datasets.load_iris()
clf = SVC()
clf.fit(iris.data, iris.target)

list(clf.predict(iris.data[:3]))

clf.fit(iris.data, iris.target_names[iris.target])

list(clf.predict(iris.data[:3]))

#Refitting and updating parameters

import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
X, y = load_iris(return_X_y=True)

clf = SVC()
clf.set_params(kernel='linear').fit(X, y)
clf.predict(X[:5])

clf.set_params(kernel='rbf').fit(X, y)
clf.predict(X[:5])

#Multiclass vs. multilabel fitting

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]

classif = OneVsRestClassifier(estimator=SVC(random_state=0))
classif.fit(X, y).predict(X)
y = LabelBinarizer().fit_transform(y)
classif.fit(X, y).predict(X)

from sklearn.preprocessing import MultiLabelBinarizer
y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
y = MultiLabelBinarizer().fit_transform(y)
classif.fit(X, y).predict(X)

#%%
#model persistence

from sklearn import svm #digunakan untuk memanggil svm di library sklearn
from sklearn import datasets # Diguankan untuk class datasets dari library sklearn
clf = svm.SVC()  # membuat variabel clf, dan memanggil class svm dan fungsi SVC
X, y = datasets.load_iris(return_X_y=True) #Mengambil dataset iris dan mengembalikan nilainya.
clf.fit(X, y)  #Perhitungan nilai label


import pickle #memanggil library pickle 
s = pickle.dumps(clf) #untuk membuat variavel s sebagai classifier
clf2 = pickle.loads(s) # variabel clf2 sebagai load(s)
clf2.predict(X[0:1]) # untuk prediksi

from joblib import dump, load #memanggil dump, load melalui library joblib
dump(clf, '1184101.joblib')   #Menyimpan model kedalam 1184101sp.joblib
clf = load('1184101.joblib')   #Memanggil model 1184101

print(clf) #menampilkan hasil model clf




 


