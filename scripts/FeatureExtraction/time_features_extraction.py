# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:32:21 2023

@author: Lorenzo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:20:36 2023

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

# VARIABILI--------------------------------------------------------------
padding = 40000


# FUNZIONI---------------------------------------------------------------

def windowing(sample):
    N = len(sample)
    w = get_window('blackmanharris', N, fftbins=False)
    return sample*w


def convert_spectrum_in_mfcc(Spectrum, sample_rate = 22050):
    mfcc = librosa.feature.mfcc(S=Spectrum, n_mfcc=13, sr = sample_rate, norm='ortho')#provare a togliere ortho
    mfcc = mfcc.mean(axis=1)#controllare
    return mfcc


def convert_signal_in_mfcc(signal, sr):

    S_trimmed = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, win_length=441, hop_length=220, center = True)
    S_db_trimmed = librosa.amplitude_to_db(S_trimmed, ref=np.max)
    mfcc = convert_spectrum_in_mfcc(S_db_trimmed)
    return mfcc

def feat_ext_from_entire_signal_freq_dom(signal, sample_rate):
    #n = len(signal)
    #freq = np.fft.rfftfreq(n, d=1/sample_rate)
    signal_ft = 20*np.log10(abs(np.fft.rfft(signal)))
    freq_dom_ft = []
    freq_dom_ft.append(np.mean(signal_ft))
    freq_dom_ft.append(np.var(signal_ft))
    
    for i in range(200):
        freq_dom_ft.append(np.mean(signal_ft[10*i:10*(i+1)]))
        freq_dom_ft.append(np.var(signal_ft[10*i:10*(i+1)]))
    for i in range(70):
        freq_dom_ft.append(np.mean(signal_ft[2000+50*i:2000+50*(i+1)]))
        freq_dom_ft.append(np.var(signal_ft[2000+50*i:2000+50*(i+1)]))
    for i in range(28):
        freq_dom_ft.append(np.mean(signal_ft[5500+150*i:5500+150*(i+1)]))
        freq_dom_ft.append(np.var(signal_ft[5500+150*i:5500+150*(i+1)]))

        
    mfcc_coeff = convert_signal_in_mfcc(signal, sample_rate)  # add mfcc coefficients
    for coeff in mfcc_coeff:
        freq_dom_ft.append(coeff)
    
    lpc_coeff = librosa.lpc(signal, order=14)
    for coeff in lpc_coeff[1:]:
        freq_dom_ft.append(coeff)
        
    return pd.Series(freq_dom_ft)

# given the signal extract mfcc, mean, std, min, max for every window
def feat_ext_from_windows(signal, sample_rate, winLen, overlap):
    features = []
    
    window_size = int(winLen*sample_rate)
    window_shift = int(window_size*overlap)
    


    # -> dim_of_pad_audio/(shift)

    for i in range(int(signal.shape[0]/(window_shift))):

        # creating the window
        

        low_bound = i*int(window_shift)
        high_bound = low_bound+window_size
        window = windowing(signal[low_bound:high_bound])

        ##short-time features##
        features.append(sum(librosa.zero_crossings(window)))
        features.append(sum(window**2))  # energia della finestra aggiunta
        features.append(np.sqrt(np.mean(window**2)))  # radice della potenza media
        features.append(window.mean())
        features.append(window.std())
        features.append(min(window))
        features.append(max(window))

        ### frequency_features ###
        window_ft = abs(np.fft.rfft(window))
        window_ft = np.where(window_ft<0.0000001, 0.0000001, window_ft)
        window_ft = 20*np.log10(window_ft)
        features.append(np.mean(window_ft))
        features.append(np.var(window_ft))
    
    features = pd.Series(features)
    return features

def col_names_generator(signal_n_samp, sample_rate):
    features_name = []

    
    for params in [(80*pow(10, -3), 0.75)]:# [(40*pow(10, -3), 0.5),(100*pow(10, -3), 0.5),(160*pow(10, -3), 0.5)]:
        for j in range(int(signal_n_samp/int(params[0]*sample_rate*params[1]))):
                features_name.append('zc su finestra'+str(j))
                features_name.append('energy su finestra'+str(j))
                features_name.append('root_mean_power su finestra'+str(j))
                features_name.append('mean su finestra'+str(j))
                features_name.append('std su finestra'+str(j))
                features_name.append('min su finestra'+str(j))
                features_name.append('max su finestra'+str(j))
                features_name.append('Media F(s) su fin Temporale'+str(j))
                features_name.append('Var F(s) su fin Temporale '+str(j))
                
    # features_name.append('Media della F(s)')
    # features_name.append('Varianza della F(s)')
    
    
    # freq = np.fft.rfftfreq(padding, d=1./22050)
    
    # for i in range(200):
    #     features_name.append('Media F(s) in:'+str(freq[10*i])+'-'+str(freq[10*(i+1)])+'Hz')
    #     features_name.append('Var F(s) in:'+str(freq[10*i])+'-'+str(freq[10*(i+1)])+'Hz')
    # for i in range(70):
    #     features_name.append('Media F(s) in:'+str(freq[2000+50*i])+'-'+str(freq[2000+50*(i+1)])+'Hz')
    #     features_name.append('Var F(s) in:'+str(freq[2000+50*i])+'-'+str(freq[2000+50*(i+1)])+'Hz')
    # for i in range(28):
    #     features_name.append('Media F(s) in:'+str(freq[5500+150*i])+'-'+str(freq[5500+150*(i+1)])+'Hz')
    #     features_name.append('Var F(s) in:'+str(freq[5500+150*i])+'-'+str(freq[5500+150*(i+1)])+'Hz')

        
    # for k in range(20):
    #     features_name.append(str(k)+' mfcc_coeff')
        
    # for k in range(14):
    #     features_name.append(str(k)+' lpc_coeff')
    return features_name 

def create_dataframe(df, start=10, end=20):
    new_df = []
    np.seterr(divide = 'ignore') 
    np.seterr(invalid = 'ignore') 
    for i in tqdm(range(start, end), file=sys.stdout):

        audio, sample_rate = librosa.load(df.iloc[i].path)
        audio_trimmed , _ = librosa.effects.trim(audio, top_db=20)
        audio_trimmed = librosa.util.fix_length(audio_trimmed, size=padding)  # pad the signal to 25000
        audio_trimmed = normalize(audio_trimmed)
        features = pd.Series(dtype='float64')
        for params in [(80*pow(10, -3), 0.75)]:# [(40*pow(10, -3), 0.5),(100*pow(10, -3), 0.5),(160*pow(10, -3), 0.5)]:
            features = pd.concat([features, feat_ext_from_windows(audio_trimmed, sample_rate, winLen = params[0], overlap = params[1])])
        #features = pd.concat([features, feat_ext_from_entire_signal_freq_dom(audio_trimmed, sample_rate)])
        new_df.append(features.values)
    
    features_name = col_names_generator(padding, sample_rate)
    new_df = pd.DataFrame(new_df, columns = features_name)

    return new_df


# CORPO-----------------------------------------------------------------
#st = time.time()
#audios = pd.DataFrame()
pathL = "dsl_data/development.csv"
df = pd.read_csv(pathL, index_col='Id')


def main():
    print('ciao')
    #values_L = ((df, 0, 1232), (df, 1232, 2*1232),
    #            (df, 2*1232, 3*1232), (df, 3*1232,  4*1232))
    #values_D = ((df, 4928, 4928+1231), (df, 4928+1231, 4928+2*1231), (df, 4928+2*1231, 4928+3*1231),(df, 4928+3*1231,  2+4928+4*1231))
    values = ((df,0, 363), (df,363, 2*363), (df,2*363, 3*363), (df, 3*363, 1455))
    #values_test = ((df,0, 10), (df,10, 20), (df,20, 30), (df, 30, 40))
    with Pool() as pool:
        res = pool.starmap(create_dataframe, values)
        pool.close()
        pool.join()
        return res

#X_train, X_test, y_train, y_test = train_test_split(   audios, y, test_size=0.20, random_state=42, shuffle=True)


if __name__ == '__main__':
    
    result = main()
    final = pd.concat([result[0], result[1], result[2], result[3]])
    final.to_csv('evaluation_set_time_features.csv')