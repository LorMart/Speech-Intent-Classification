to_delete = [25, 4125, 8223, 36, 8228, 8229, 8230, 2096, 4148, 4149, 55, 4152, 6201, 8247, 2123, 75, 4172, 4173, 2131, 4183,
             8286, 4191, 108, 6264, 121, 8316, 8318, 130, 134, 2189, 8342, 6297, 4261, 6321, 4279, 184, 2232, 8378, 200, 2253,
             8405, 2262, 8411, 6364, 8420, 6377, 8430, 8432, 8434, 8438, 8449, 8450, 2306, 4358, 263, 8456, 8457, 6406, 8468,
             8473, 2334, 4394, 332, 335, 8527, 4433, 8528, 341, 6499, 8547, 6504, 4460, 6509, 2419, 380, 8576, 4521, 8619, 4524,
             8620, 4531, 6599, 8648, 8654, 471, 2520, 6616, 484, 8677, 493, 8696, 2556, 8703, 4608, 8710, 8719, 4633, 539, 8732,
             2593, 6691, 2597, 8742, 6699, 8747, 8751, 6704, 8757, 568, 569, 6717, 8767, 6723, 4677, 4690, 2648, 8793, 2650,
             4699, 8800, 616, 4715, 6765, 8818, 4723, 628, 2680, 6790, 8838, 650, 8845, 8849, 6810, 4768, 2738, 8890, 4802,
             8898, 8901, 2765, 2777, 2781, 8925, 746, 8946, 4853, 8949, 8953, 8958, 8959, 6911, 4866, 8963, 2826, 8971, 2835,
             8980, 8987, 8996, 8999, 9014, 6969, 9017, 831, 2879, 9026, 4931, 2888, 844, 9036, 9041, 9042, 6995, 853, 9054,
             2933, 4985, 2940, 9098, 2958, 9107, 9126, 9128, 2985, 939, 9131, 5038, 7096, 9149, 3008, 9158, 7115, 9163, 9164,
             7123, 9179, 9202, 5108, 7158, 9207, 3066, 3067, 9213, 7167, 3079, 3081, 9225, 5136, 9232, 9234, 7188, 3093, 9240,
             9248, 3109, 3112, 9258, 5167, 7219, 9269, 9270, 9275, 9279, 9290, 9305, 9309, 7265, 5225, 3179, 3187, 3190, 5238,
             1144, 5240, 9338, 5244, 5246, 3216, 3219, 9363, 1174, 9369, 7330, 5285, 3239, 3240, 3243, 3244, 3245, 5292, 9395,
             3252, 3253, 9400, 3259, 3266, 3267, 3270, 7367, 3272, 5321, 3275, 3277, 1230, 3279, 5326, 3282, 3287, 7383, 3294,
             3299, 3300, 3301, 3304, 9449, 3306, 7404, 9454, 3322, 7419, 9467, 1283, 3331, 7431, 1288, 9479, 1294, 9486, 9489,
             3350, 9499, 9502, 9503, 9508, 9510, 9524, 9525, 3396, 5452, 7502, 9556, 9559, 1374, 5472, 9568, 5476, 1392, 5497,
             9593, 7548, 9596, 5512, 9616, 3474, 3480, 5533, 9630, 7588, 9639, 7600, 9663, 7617, 9665, 5574, 1481, 5580, 5584,
             9687, 9688, 5595, 1506, 1509, 9704, 9724, 5629, 5636, 7684, 3592, 3597, 1555, 3609, 7706, 1574, 9784, 1597, 3657,
             1616, 9829, 1639, 9832, 9833, 1643, 7792, 3750, 7871, 3784, 3799, 1753, 3807, 5859, 5866, 3823, 3827, 5881, 3839,
             7950, 5906, 7962, 7972, 7975, 5930, 7989, 3896, 7993, 7994, 3899, 1852, 1854, 7999, 5956, 8013, 5972, 8022, 3929,
             3934, 5983, 3952, 1913, 1936, 8085, 6047, 4000, 8095, 6050, 8112, 1970, 1981, 1993, 8140, 8141, 4047, 2006, 2008,
             8154, 2011, 8159, 2022, 2025, 2026, 4073, 4082, 2039, 4089, 2042]

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
        features.append(sum(window**2))  # energia della finestra aggiunta
        features.append(np.sqrt(np.mean(window**2)))  # radice della potenza media
        features.append(window.mean())
        features.append(window.std())
        features.append(min(window))
        features.append(max(window))

    
    features = pd.Series(features)
    return features

def col_names_generator(signal_n_samp, sample_rate):
    features_name = []

    
    for params in [(40*pow(10, -3), 0.5),(40*pow(10, -3), 0.75),(80*pow(10, -3), 0.5),(80*pow(10, -3), 0.75),(120*pow(10, -3), 0.5),(120*pow(10, -3), 0.75)]:
        for j in range(int(signal_n_samp/int(params[0]*sample_rate*params[1]))):
                features_name.append('energy su finestra'+str(j))
                features_name.append('root_mean_power su finestra'+str(j))
                features_name.append('mean su finestra'+str(j))
                features_name.append('std su finestra'+str(j))
                features_name.append('min su finestra'+str(j))
                features_name.append('max su finestra'+str(j))
                
    return features_name 

def create_dataframe(df, start=10, end=20):
    new_df = [] 
    for i in tqdm(range(start, end), file=sys.stdout):

        audio, sample_rate = librosa.load(df.iloc[i].path)
        audio_trimmed , _ = librosa.effects.trim(audio, top_db=25)
        audio_trimmed = librosa.util.fix_length(audio_trimmed, size=padding)  # pad the signal to 40000
        audio_trimmed = normalize(audio_trimmed)
        features = pd.Series(dtype='float64')
        for params in [(40*pow(10, -3), 0.5),(40*pow(10, -3), 0.75),(80*pow(10, -3), 0.5),(80*pow(10, -3), 0.75),(120*pow(10, -3), 0.5),(120*pow(10, -3), 0.75)]:
            features = pd.concat([features, feat_ext_from_windows(audio_trimmed, sample_rate, winLen = params[0], overlap = params[1])])
        new_df.append(features.values)
    
    features_name = col_names_generator(padding, sample_rate)
    new_df = pd.DataFrame(new_df, columns = features_name)

    return new_df


# CORPO-----------------------------------------------------------------
#st = time.time()
#audios = pd.DataFrame()
pathL = "dsl_data/development.csv"
df = pd.read_csv(pathL, index_col='Id')
df.drop(index = to_delete,inplace=True)


def main():
    print('ciao')
    #values_L = ((df, 0, 1232), (df, 1232, 2*1232),
    #            (df, 2*1232, 3*1232), (df, 3*1232,  4*1232))
    #values_D = ((df, 4928, 4928+1231), (df, 4928+1231, 4928+2*1231), (df, 4928+2*1231, 4928+3*1231),(df, 4928+3*1231,  2+4928+4*1231))
    values = ((df,0, 2360), (df,2360, 2*2360), (df,2*2360, 3*2360), (df, 3*2360, 4*2360))
    with Pool() as pool:
        res = pool.starmap(create_dataframe, values)
        pool.close()
        pool.join()
        return res

#X_train, X_test, y_train, y_test = train_test_split(   audios, y, test_size=0.20, random_state=42, shuffle=True)


if __name__ == '__main__':
    
    result = main()
    final = pd.concat([result[0], result[1], result[2], result[3]])
    final.to_csv('all_windows_blackman.csv')