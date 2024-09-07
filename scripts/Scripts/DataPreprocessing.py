# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 21:43:21 2023

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
import matplotlib.pyplot as mpl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from librosa.feature import spectral_flatness


pathL = r"C:\Users\Lorenzo\Desktop\Progetto DSL\dsl_data\development.csv"
df = pd.read_csv(pathL, index_col='Id')

lengths = []
pathD = r"C:\Users\Lorenzo\Desktop\Progetto DSL\\"
audios = []
spec_list = []

for i in tqdm(range(df.shape[0])):
    audio, sample_rate = librosa.load(pathD + df.iloc[i].path)
    audios.append(audio)
    lengths.append(len(audio))
    
new_df = pd.concat([df, pd.DataFrame(lengths, columns = ['lengths'])], axis = 1)
outLL_I = new_df.index[new_df['lengths'] == 441000.].tolist()
outLL_signals=[]



#%%


mpl.hist(lengths, bins=500)
mpl.gca().set(title='Original Signals Length', ylabel='Frequency', xlabel='Length in samples');

for i in outLL_I:
    #outLL_signals.append(audios[i]) #salva gli ouliers in un vettore
    #print(np.max(np.nonzero(audios[i].nonzero()))) #lunghezze Senza padding
    lengths[i] = np.max(np.nonzero(audios[i].nonzero()))
    audios[i] = (audios[i][:np.max(np.nonzero(audios[i].nonzero()))])
    

mpl.hist(lengths, bins=500)
mpl.gca().set(title='Signals Length After removing zeros', ylabel='Frequency', xlabel='Length in samples');

lengths_after_removing_outliers = lengths;

spec_list=[]

for i in range(len(audios)):
    audios[i], _ = librosa.effects.trim(audios[i], top_db=25)
    lengths[i] = len(audios[i])
    
for i in tqdm(range(len(audios))):
    spec_list.append(spectral_flatness(audios[i]))

spec_list_mean = []
for i in tqdm(range(len(audios))):
    spec_list_mean.append(spec_list[i][0].mean())
        
#Funzione per riprodurre i segnali

mpl.hist(lengths, bins=1000, range = (0, 60000))
mpl.gca().set(title='Signals Length After Trimming', ylabel='Frequency', xlabel='Length in samples');

lengths_npa_S = np.array(lengths)/22050
mpl.hist(lengths_npa_S, bins=1000)
mpl.gca().set(title='Signals Length After Trimming inSec', ylabel='Frequency', xlabel='Length in samples');


#%% 



for i in tqdm(range(len(audios))):
    spec_list.append(spectral_flatness(audios[i]))

spec_list_mean = []
for i in tqdm(range(len(audios))):
    spec_list_mean.append(spec_list[i][0].mean())
    
mpl.hist(spec_list_mean, bins=50)
mpl.gca().set(title='Signal SP flatness', ylabel='Frequency', xlabel='Length in samples');

#%% 
#plot dei segnali con SF < di una soglia
basso_spec = [[i, spec_list_mean[i]]  for i in range(len(spec_list_mean)) if spec_list_mean[i] > 0.015]
to_plot_basso_spec = []

for i in range(len(basso_spec)):
    to_plot_basso_spec.append(basso_spec[i][1])
mpl.hist(to_plot_basso_spec, bins=500)
mpl.gca().set(title='Signal SP flatness', ylabel='Frequency', xlabel='Length in samples');

#%%
# def signaltonoise(a, axis=0, ddof=0):
#     a = np.asanyarray(a)
#     m = a.mean(axis)
#     sd = a.std(axis=axis, ddof=ddof)
#     return np.where(sd == 0, 0, m/sd)

# snr_vett = []

# for i in range(len(audios)):
#     snr_vett.append(signaltonoise(audios[i], axis = 0, ddof=0))
    

# mpl.hist(snr_vett, bins=500)
# mpl.gca().set(title='Signal SP flatness', ylabel='Frequency', xlabel='Length in samples');


# alto_snr = [[i, snr_vett[i]] for i in range(len(snr_vett)) if abs(snr_vett[i]) > 0.05]

lengths_npa_S = np.array(lengths)/22050
spec = 0.01;
tempo = 0;
for i in range(100):
    tempo = tempo + 0.01
    spec = 0.000;
    for j in range(1000):
        spec = spec + 0.001
to_delete = list( set( ((lengths_npa_S < 0.052).nonzero()[0]).tolist() + ((lengths_npa_S > 4).nonzero()[0]).tolist()))
basso_spec_indexes = [basso_spec[i][0] for i in range(len(basso_spec)) if basso_spec[i][1] > 0.015]
to_delete = list( set( basso_spec_indexes + to_delete))
        if(len(to_delete) == 414):
            print('\nBeccato!\n:')
            print(tempo)
            print('\n:')
            print(spec)

       
    
#%%
stored = []

ti= 0.0;
tf = 3.5;
spec = 0.0;
for i in tqdm(range(100)):
    ti = ti+0.01
    for j in tqdm(range(300)):
        tf = tf + 0.01
        for k in range(50):
            spec = spec+0.001
            to_delete = list( set( ((lengths_npa_S < 1).nonzero()[0]).tolist() + ((lengths_npa_S > 4.5).nonzero()[0]).tolist()))
            basso_spec_indexes = [basso_spec[i][0] for i in range(len(basso_spec)) if basso_spec[i][1] > 0.015]
            to_delete = list( set( basso_spec_indexes + to_delete))
            if(len(to_delete)> 414 & len(to_delete) < 420):
               
                to_delete.append(spec)
                to_delete.append(tf)
                to_delete.append(ti)
                stored.append(to_delete)
                
        spec = 0.0;
    tf = 3.5;

#%%
df.drop(index = to_delete, inplace = True)
for i in to_delete:
    audios.pop(i)
    
#%%

def convert_spectrum_in_mfcc(Spectrum, sample_rate = 22050):
    mfcc = librosa.feature.mfcc(S=Spectrum, n_mfcc=13, sr = sample_rate)#provare a togliere ortho
    mfcc = mfcc.mean(axis=1)#controllare
    return mfcc
def convert_signal_in_mfcc(signal, sr, win_length, hop_length):
    S_trimmed = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, win_length=win_length, hop_length=hop_length, center = True)
    S_db_trimmed = librosa.amplitude_to_db(S_trimmed, ref=np.max)
    mfcc = convert_spectrum_in_mfcc(S_db_trimmed)
    return mfcc

mfcc_coeff = []
for i in tqdm(range(len(audios))):
    mfcc_coeff.append(convert_signal_in_mfcc(audios[i], 22050, win_length = 441, hop_length=110))
mfcc_df = pd.DataFrame(mfcc_coeff)

mfcc_coeff_2 = []
# for i in tqdm(range(len(audios))):
#     mfcc_coeff_2.append(convert_signal_in_mfcc(audios[i], 22050, win_length = 3*441, hop_length=330))
# mfcc_2_df = pd.DataFrame(mfcc_coeff_2)

#%%
lpc_coeff = []

for i in tqdm(range(len(audios))):
    lpc = librosa.lpc((audios[i]), order=14)
    lpc_coeff.append(lpc[1:])
    
lpc_df = pd.DataFrame(lpc_coeff)

y = (df.loc[:, 'action'])+(df.loc[:, 'object'])

#%%
final_df = pd.concat([lpc_df, mfcc_df], axis = 1)

#%%
from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification
from sklearn.preprocessing import OrdinalEncoder

final_df = pd.concat([lpc_df, mfcc_df], axis = 1)
final_df = final_df.fillna(0)

enc = OrdinalEncoder()
y = enc.fit_transform(pd.DataFrame(y))


undersample = RandomUnderSampler(sampling_strategy='majority')
X_over, y_over = undersample.fit_resample(final_df, y)

X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.10, random_state=42)




#%%


def windowing(sample):
    N = len(sample)
    w = get_window('blackmanharris', N, fftbins=False)
    return sample*w
def feat_ext_from_windows(signal, sample_rate, winLen=40*pow(10, -3), overlap=0.50):
    features = []
    
    window_size = int(winLen*sample_rate)
    window_shift = int(window_size*overlap)
    


    # -> dim_of_pad_audio/(shift)

    for i in range(int(signal.shape[0]/(window_shift))):

        # creating the window
        
        low_bound = i*int(window_shift)
        high_bound = low_bound+window_size
        window = signal[low_bound:high_bound]


        ##short-time features##
        features.append(sum(librosa.zero_crossings(window)))
        features.append(sum(window**2))  # energia della finestra aggiunta
        features.append(np.sqrt(np.mean(window**2)))  # radice della potenza media
        features.append(window.mean())
        features.append(window.std())
        features.append(min(window))
        features.append(max(window))

        window = windowing(signal[low_bound:high_bound])
        
        ### frequency_features ###
        window_ft = abs(np.fft.rfft(window))
        window_ft = np.where(window_ft<0.0000001, 0.0000001, window_ft)
        window_ft = 20*np.log10(window_ft)
        features.append(np.mean(window_ft))
        features.append(np.var(window_ft))
    
    # features = pd.Series(features)
    return features
    
frame_features = []
for i in (range(len(audios))):
    frame_features.append(feat_ext_from_windows(audios[i], 22050, 882, 0.75))
    
#%%
# X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)
clf = RandomForestClassifier(n_estimators = 5000, criterion = 'entropy', min_samples_split=4, random_state=0, n_jobs = -1)
clf.fit(X_train, y_train.reshape(y_train.shape[0]))
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))

#%%
#n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None
param_grid = [{'n_estimators': [5000, 700, 1000], 'criterion': ['gini', 'entropy'], 'min_samples_split':[8, 4], 'max_features':['sqrt', 'log2']}]

from sklearn.model_selection import RandomizedSearchCV


clf = RandomizedSearchCV(RandomForestClassifier(), param_grid,n_jobs = -1, refit = True, verbose=3, n_iter=5)
clf.fit(X_train, y_train.reshape(y_train.shape[0]))


#%%
def windowing(sample):
    N = len(sample)
    w = get_window('blackmanharris', N, fftbins=False)
    return sample*w
def feat_ext_from_windows(signal, sample_rate, winLen=40*pow(10, -3), overlap=0.50):
    features = []
    
    window_size = int(winLen*sample_rate)
    window_shift = int(window_size*overlap)
    


    # -> dim_of_pad_audio/(shift)

    for i in range(int(signal.shape[0]/(window_shift))):

        # creating the window
        
        low_bound = i*int(window_shift)
        high_bound = low_bound+window_size
        window = signal[low_bound:high_bound]


        ##short-time features##
        features.append(sum(librosa.zero_crossings(window)))
        features.append(sum(window**2))  # energia della finestra aggiunta
        features.append(np.sqrt(np.mean(window**2)))  # radice della potenza media
        features.append(window.mean())
        features.append(window.std())
        features.append(min(window))
        features.append(max(window))

        window = windowing(signal[low_bound:high_bound])
        
        ### frequency_features ###
        window_ft = abs(np.fft.rfft(window))
        window_ft = np.where(window_ft<0.0000001, 0.0000001, window_ft)
        window_ft = 20*np.log10(window_ft)
        features.append(np.mean(window_ft))
        features.append(np.var(window_ft))
    
    # features = pd.Series(features)
    return features

