import pickle
import numpy as np
import sys
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


np.set_printoptions(threshold=sys.maxsize)

pickle_in = open("train_image.pkl","rb")
Train_data = pickle.load(pickle_in)
X_train = np.array(Train_data)

pickle_in = open("train_label.pkl","rb")
Train_labels = pickle.load(pickle_in)
Y_train = np.array(Train_labels)


pickle_in = open("test_image.pkl","rb")
Test_data = pickle.load(pickle_in)
X_test = np.array(Test_data)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)


neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(X_train, Y_train) 
y_pred=neigh.predict(X_val)

print (accuracy_score(Y_val, y_pred))
