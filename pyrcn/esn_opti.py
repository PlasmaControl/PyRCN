import numpy as np
import pandas as pd
import os
import librosa
from statistics import mean
from time import process_time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.base import clone


def esn_train_step(base_esn, params, audio, vocalInfo, vowels):
    esn = clone(base_esn)  # leeres Basis_ESN kopieren
    esn.set_params(**params)
    # time_train_start = process_time()
    with tqdm(total=len(audio) - 1) as pbar:  # Fortschrittsbalken für folgende Schleife
        for y, vowel in zip(audio[:-1], vocalInfo[:-1]):  # Schleife über alle audio-Werte, bis auf den letzten
            X = np.zeros(shape=(len(y), 5),
                         dtype="int")  # X-Matrix anlegen mit der Länge des Audiosignals und Breite=5 (Anzahl der Vokale)
            X[:, vowels[vowel]] = 1  # one-hot-vactor für jeden y-Wert -> Vokalzuordnug = 1, Rest der Vokale =0
            esn.partial_fit(X=X, y=y.reshape(-1, 1), postpone_inverse=True)  # ESN partial fit, ohne inverse
            pbar.update(1)
    # time_train_end = process_time()

    y = audio[-1]  # letzter Audio-file wird geladen
    X = np.zeros(shape=(len(y), 5), dtype="int")
    X[:, vowels[vocalInfo[-1]]] = 1
    esn.partial_fit(X=X, y=y.reshape(-1, 1),
                    postpone_inverse=False)  # esn weiter fitten mit letztem audio und der inversen
    return esn


def esn_train_impulse(base_esn, params, audio, vocalInfo, vowels):
    esn = clone(base_esn)  # leeres Basis_ESN kopieren
    esn.set_params(**params)
    # time_train_start = process_time()
    with tqdm(total=len(audio) - 1) as pbar:  # Fortschrittsbalken für folgende Schleife
        for y, vowel in zip(audio[:-1], vocalInfo[:-1]):  # Schleife über alle audio-Werte, bis auf den letzten
            X = np.zeros(shape=(len(y), 5),
                         dtype="int")  # X-Matrix anlegen mit der Länge des Audiosignals und Breite=5 (Anzahl der Vokale)
            X[0, vowels[vowel]] = 1  # one-hot-vactor für jeden y-Wert -> Vokalzuordnug = 1, Rest der Vokale =0
            esn.partial_fit(X=X, y=y.reshape(-1, 1), postpone_inverse=True)  # ESN partial fit, ohne inverse
            pbar.update(1)
    # time_train_end = process_time()

    y = audio[-1]  # letzter Audio-file wird geladen
    X = np.zeros(shape=(len(y), 5), dtype="int")
    X[0, vowels[vocalInfo[-1]]] = 1
    esn.partial_fit(X=X, y=y.reshape(-1, 1),
                    postpone_inverse=False)  # esn weiter fitten mit letztem audio und der inversen
    return esn


def esn_optimize_impulse(base_esn, params, audio, vocalInfo, vowels, csv_file, spectral_params={'fmin': 200,
                                                                                                'fmax': 3700,
                                                                                                'sr' : 16000,
                                                                                                'n_mels': 15}):
    time_train_start = process_time()
    esn = esn_train_impulse(base_esn, params, audio, vocalInfo, vowels)
    time_train_end = process_time()

    err_train = []
    err_train_mel = []
    err_train_mel_log = []
    err_train_mfcc = []
    time_vali_start = process_time()
    for y, vowel in zip(audio, vocalInfo):        # Testen des trainierten ESN und Fehlerbestimmung
        X = np.zeros(shape=(len(y), 5), dtype="int")
        X[0, vowels[vowel]] = 1
        #X[-1,,vowels[vowel]] = -1
        y_pred = esn.predict(X=X)
        err_train.append(mean_squared_error(y, y_pred))  # Fehler für jeden Wert im Audiosignal
        y_mel, y_pred_mel = melspec_calc(y, y_pred, spectral_params)  # mel-coef calculate for y and y_pred
        err_train_mel_log.append(mean_squared_error(np.log10(y_mel), np.log10(y_pred_mel)))  # mse of log(mel-coeff)
        err_train_mel.append(mean_squared_error(y_mel, y_pred_mel))  # mse of mel coeffi
        y_mfcc, y_pred_mfcc = mfcc_calc(y, y_pred, spectral_params)
        err_train_mfcc.append(mean_squared_error(y_mfcc, y_pred_mfcc))
    time_vali_end= process_time()

    # save prams and error in csv-file
    df_Params = pd.DataFrame()
    df_Params = df_Params.append(params, ignore_index=True)
    df_Params['err_train'] = mean(err_train)
    df_Params['err_train_mel'] = mean(err_train_mel)
    df_Params['err_train_mel_log'] = mean(err_train_mel_log)
    df_Params['err_train_mfcc'] = mean(err_train_mfcc)
    df_Params['time_train'] = time_train_end - time_train_start
    df_Params['time_vali'] = time_vali_end - time_vali_start

    if os.path.exists(csv_file):
        df_Params.to_csv(csv_file, mode='a', header=False)
    else:
        df_Params.to_csv(csv_file, mode='w', header=True)
    return df_Params


def esn_optimize_step(base_esn, params, audio, vocalInfo, vowels, csv_file, spectral_params={'fmin': 200,
                                                                                             'fmax': 3700,
                                                                                             'sr' : 16000,
                                                                                             'n_mels': 15}):
    time_train_start = process_time()
    esn =esn_train_step(base_esn, params, audio, vocalInfo, vowels)
    time_train_end = process_time()

    err_train = []
    err_train_mel = []
    err_train_mel_log = []
    err_train_mfcc = []
    time_vali_start=process_time()
    for y, vowel in zip(audio, vocalInfo):        # Testen des trainierten ESN und Fehlerbestimmung
        X = np.zeros(shape=(len(y), 5), dtype="int")
        X[:, vowels[vowel]] = 1
        # X[-1,vowels[vowel]] = -1
        y_pred = esn.predict(X=X)
        err_train.append(mean_squared_error(y, y_pred))  # Fehler für jeden Wert im Audiosignal
        y_mel, y_pred_mel = melspec_calc(y, y_pred, spectral_params)    # mel-coef calculate for y and y_pred
        err_train_mel_log.append(mean_squared_error(np.log10(y_mel), np.log10(y_pred_mel)))  # mse of log(mel-coeff)
        err_train_mel.append(mean_squared_error(y_mel,y_pred_mel))  # mse of mel coeffi
        y_mfcc, y_pred_mfcc = mfcc_calc(y, y_pred, spectral_params)
        err_train_mfcc.append(mean_squared_error(y_mfcc, y_pred_mfcc))

    time_vali_end = process_time()

    # save prams and error in csv-file
    df_Params = pd.DataFrame()
    df_Params = df_Params.append(params, ignore_index=True)

    df_Params['err_train'] = mean(err_train)
    df_Params['err_train_mel'] = mean(err_train_mel)
    df_Params['err_train_mel_log'] = mean(err_train_mel_log)
    df_Params['err_train_mfcc'] = mean(err_train_mfcc)
    df_Params['time_train'] = time_train_end - time_train_start
    df_Params['time_vali'] = time_vali_end - time_vali_start

    if os.path.exists(csv_file):
        df_Params.to_csv(csv_file, mode='a', header=False)
    else:
        df_Params.to_csv(csv_file, mode='w', header=True)
    return df_Params


def melspec_calc(y, y_pred, spectral_params): # Melspectrum
    winLength = int(spectral_params['sr'] * 0.032)
    hopLength = int(spectral_params['sr'] * 0.01)
    y_mel = librosa.feature.melspectrogram(y=y, win_length=winLength, hop_length=hopLength, window=np.hamming(winLength),
                                           n_fft=winLength, **spectral_params)
    y_pred_mel = librosa.feature.melspectrogram(y=y_pred, win_length= winLength, hop_length= hopLength, window=np.hamming(winLength),
                                                n_fft=winLength, **spectral_params)
    return y_mel, y_pred_mel


def mfcc_calc(y,y_pred, spectral_params):
    y_mel, y_pred_mel = melspec_calc(y, y_pred, spectral_params)
    y_mel_log = np.log10(y_mel)
    y_pred_mel_log = np.log10(y_pred_mel)
    y_mfcc = librosa.feature.mfcc(S=y_mel_log, **spectral_params)
    y_pred_mfcc = librosa.feature.mfcc(S=y_pred_mel_log, **spectral_params)
    return y_mfcc, y_pred_mfcc


def esn_impulse_train_seq(audio, vocalInfo, vowels,esn):
    X = np.empty(shape=(len(audio),), dtype=object)
    y = np.empty(shape=(len(audio),), dtype=object)
    for k, (y_true, vowel) in enumerate(zip(audio, vocalInfo)):  # Schleife über alle audio-Werte, bis auf den letzten
        X_true = np.zeros(shape=(len(y_true), 5),
                          dtype="int")  # X-Matrix anlegen mit der Länge des Audiosignals und Breite=5 (Anzahl der Vokale)
        X_true[0, vowels[vowel]] = 1  # one-hot-vactor für jeden y-Wert -> Vokalzuordnug = 1, Rest der Vokale =0
        X[k] = X_true
        y[k] = y_true.reshape(-1, 1)
    esn.fit(X,y,n_jobs=24) # njobs =  zahl der verwendeten kerne