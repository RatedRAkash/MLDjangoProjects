import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('../data/Social_Network_Ads.csv')
dataset["Gender"] = 1 if "Male" else 2
X = dataset.iloc[:, 1:4]
y = dataset.iloc[:, 4]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1000)

# StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

filename = '../trained_models/TrainedScaler.sav'
pickle.dump(scaler, open(filename, 'wb'))


# SVC Prediction Model
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

filename = '../trained_models/PredictionModel.sav'
pickle.dump(classifier, open(filename, 'wb'))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)