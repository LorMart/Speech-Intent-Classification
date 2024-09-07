# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 00:15:13 2023

@author: Lorenzo
"""


import pandas as pd
import numpy as np
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from librosa.util import normalize
from scipy.signal import freqz
from scipy.signal import butter, lfilter
import time
import scipy.stats
from tqdm import tqdm
from scipy.signal import get_window
from multiprocessing import Pool, cpu_count
import sys
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
pathL = "dsl_data/development.csv";

X = pd.read_csv('test.csv')
df = pd.read_csv(pathL, index_col = 'Id')
X.drop(columns=['Unnamed: 0'], inplace = True)
y = df['action'] + df['object']
df.drop(columns=['action','object','path', 'speakerId'], inplace=True)
df_to_concat = pd.get_dummies(df)
X.drop(index=9350, inplace=True)
y.drop(index=9350, inplace=True)

corr_mat = X.corr()

#sns.heatmap(corr_mat)
#plt.figure()

sns.heatmap(corr_mat.iloc[:1200,:1200])


#energy su finestra1.1':'min su finestra1.1','mean su finestra7':'max su finestra8