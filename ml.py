from sklearn.svm import LinearSVC
import numpy as np
import pickle
from sklearn.externals import joblib

# See examples online for more info
clf = LinearSVC()  # creates generic linear SVC
mfccs = np.load("robin.npy")
mfccs = np.vstack((mfccs, np.load("chickadee.npy")))
mfccs = np.vstack((mfccs, np.load("bluejay.npy")))
mfccs = np.vstack((mfccs, np.load("cardinal.npy")))  # loads in preprocessed numpy arrays and concatenates
X = mfccs
Y = np.empty([40])
Y[0:10] = 1  # "American Robin"
Y[10:20] = 2  # "Black-Capped Chickadee"
Y[20:30] = 3  # "Blue Jay"
Y[30:40] = 4  # "Northern Cardinal"  # assigns labels
clf.fit(X, Y)
# testd = np.load("testdata.npy")
# f = clf.predict(testd)  # f can be output using label values above to give bird species to the user
# g = 2  # random thing to be able to place break point here
# joblib.dump(clf, 'TrainedClassifier.pkl')  # exports classifier to file, can also try just using pickle
