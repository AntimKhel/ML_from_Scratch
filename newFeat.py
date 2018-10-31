#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

diab = pd.read_csv('diabetes.csv')
diab.dropna(axis=0, how='any')
diab.isnull().sum()
newFeat = diab['Glucose'] * diab['BloodPressure']
newFeat2 = diab['SkinThickness'] * diab['Pregnancies'] * 0.5
idx = 0
diab.insert(loc=8, column='Sugar', value=newFeat)
diab.insert(loc=diab.columns.get_loc('Outcome'), column='SkinFragile',
            value=newFeat2)
print diab.head(5)

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

outcome = diab['Outcome']
data = diab[diab.columns[:10]]
(train, test) = train_test_split(diab, test_size=0.25, random_state=0,
                                 stratify=diab['Outcome'])  # stratify the outcome
train_X = train[train.columns[:10]]
test_X = test[test.columns[:10]]
train_Y = train['Outcome']
test_Y = test['Outcome']

types = ['rbf', 'linear']
for i in types:
    model = svm.SVC(kernel=i)
    model.fit(train_X, train_Y)
    prediction = model.predict(test_X)
    print ('Accuracy for SVM kernel=', i, 'is',
           metrics.accuracy_score(prediction, test_Y))

model = LogisticRegression()
model.fit(train_X, train_Y)
prediction = model.predict(test_X)
print ('The accuracy of the Logistic Regression is',
       metrics.accuracy_score(prediction, test_Y))
