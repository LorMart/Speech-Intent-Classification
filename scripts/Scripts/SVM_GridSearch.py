# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 23:25:55 2023

@author: Lorenzo
"""

import pandas as pd
import numpy as np
import librosa
import librosa.display
from librosa.util import normalize
from tqdm import tqdm
from scipy.signal import get_window
from multiprocessing import Pool
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , balanced_accuracy_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

pathL = r"dsl_data\development.csv"
df = pd.read_csv(pathL)

features = pd.read_csv(r"test.csv")

features.drop(columns = ['Unnamed: 0'], inplace=True)
features.drop(index=9350, inplace = True)
df.drop(index=9350, inplace = True)

to_append = pd.concat([df.loc[:, 'speakerId'], df.loc[:, 'Self-reported fluency level ':'ageRange']], axis = 1)
#df[:, ['speakerId', df.columns['Self-reported fluency level':'ageRange']]]
le = LabelEncoder()
for col in to_append.select_dtypes(include='O').columns:
    to_append[col]=le.fit_transform(to_append[col])
    
features = pd.concat([features, to_append], axis = 1)

y = df.loc[:, 'action']+df.loc[:, 'object']

scaler = StandardScaler()
X = scaler.fit_transform(features)

[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# param_grid = [
#               {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree':[3, 5, 7], 'gamma':['scale','auto'], 'class_weight': ['balanced', 'None'], 'random_state': [42], 'probability': [True, False], 'verbose':[True]},
#               {'C': [1, 10, 100, 1000], 'kernel': ['rbf'],'gamma':['scale','auto'], 'class_weight': ['balanced', 'None'], 'random_state': [42], 'probability': [True, False], 'verbose':[True]},
#               {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'],'gamma':['scale','auto'], 'class_weight': ['balanced', 'None'], 'random_state': [42], 'probability': [True, False], 'verbose':[True]}]

param_grid = [{'C': [10, 100, 1000], 'kernel': ['sigmoid'],'gamma':['scale','auto', 0.001], 'class_weight': ['balanced', 'none'], 'random_state': [42], 'probability': [True], 'verbose':[True]} ]

# svc = SVC(random_state= 42,
# probability= False,
# kernel= 'rbf',
# gamma= 'scale',
# class_weight='balanced',
# C= 1000)



clf = RandomizedSearchCV(SVC(), param_grid,n_jobs = -1, refit = True, verbose=3, n_iter=10)

clf.fit(X_train, y_train)

#svc.fit(X_train, y_train)
#y_pred = svc.predict(X_test)
#accuracy_score(y_test, y_pred)

