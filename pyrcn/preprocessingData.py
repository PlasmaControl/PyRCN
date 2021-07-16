import pandas as pd
import numpy as np
from scipy.io.wavfile import read as read_wav
import librosa
import os
from sklearn.preprocessing import MinMaxScaler

def preprocessing_audio(data_info_path, audio_path):
    sampleRate = 16000              # Ziel Samplefrequence in Hz
    cutAudio = 0.3                  # je am Anfang/Ende abgeschnittener Audioanteil  (um Pause zu entfernen)
    lengthAudio = 1 - 2 * cutAudio  # gesamtlänge der Vokaldatei in Prozent/100
    audio = []      # Liste für Audiodateien
    vocalInfo = []  # Liste für Vokalinfo
    _, _, filenames = next(os.walk(data_info_path))         # filenames aus ordner mit .csv entnehmen


    # Audio und zugehörigen Vokal in Listen speichern
    for i in range(len(filenames)):
        name = filenames[i]
        data_info = pd.read_csv(data_info_path + "/" + name)  # .csv für Audio einlesen
        timemarkBeginn = data_info['Beginn']  # Inhalt von .csv aufteilen
        timemarkEnde = data_info['Ende']
        vokal = data_info['Vokal']

        nameAudio = name.replace("csv", "wav")  # Name des Audiofiles erstellen
        pathAudio = audio_path + "/" + nameAudio  # Pfad des Ausiofiles
        Fs, _ = read_wav(pathAudio)  # SampleRate des Origianl-Audios

        for i in range(len(timemarkBeginn)):
            timemark1 = timemarkBeginn[i]
            timemark2 = timemarkEnde[i]
            vocalLength = (timemark2 - timemark1) / Fs  # Vokallänge mit Pause in Sekunden
            offset1 = (timemark1 / Fs + cutAudio * vocalLength)  # in Sekunden, start des Vokals in Sekunden in wav-file
            dauer = vocalLength * lengthAudio  # in Sekunden, % vorne und hinten abschneiden um Pause abzutrennen

            y, _ = librosa.load(path=pathAudio, sr=sampleRate, mono=True, offset=offset1,
                                duration=dauer)  # , dtype=<class 'numpy.float32'>, res_type='kaiser_best')

            y = librosa.util.normalize(y)
            audio.append(y)
            vocalInfo.append(vokal[i])
    return audio, vocalInfo, sampleRate


def preprocessing_audio_fb(data_info_path, audio_path):     # unterschied: Normierung des Audiosignals, |y|<1 um tanh im esn zu verwenden
    sampleRate = 16000              # Ziel Samplefrequence in Hz
    cutAudio = 0.3                  # je am Anfang/Ende abgeschnittener Audioanteil  (um Pause zu entfernen)
    lengthAudio = 1 - 2 * cutAudio  # gesamtlänge der Vokaldatei in Prozent/100
    audio = []      # Liste für Audiodateien
    vocalInfo = []  # Liste für Vokalinfo
    _, _, filenames = next(os.walk(data_info_path))         # filenames aus ordner mit .csv entnehmen


    # Audio und zugehörigen Vokal in Listen speichern
    for i in range(len(filenames)):
        scaler = MinMaxScaler(feature_range=(0,0.999))
        name = filenames[i]
        data_info = pd.read_csv(data_info_path + "/" + name)  # .csv für Audio einlesen
        timemarkBeginn = data_info['Beginn']  # Inhalt von .csv aufteilen
        timemarkEnde = data_info['Ende']
        vokal = data_info['Vokal']

        nameAudio = name.replace("csv", "wav")  # Name des Audiofiles erstellen
        pathAudio = audio_path + "/" + nameAudio  # Pfad des Ausiofiles
        Fs, _ = read_wav(pathAudio)  # SampleRate des Origianl-Audios

        for i in range(len(timemarkBeginn)):
            timemark1 = timemarkBeginn[i]
            timemark2 = timemarkEnde[i]
            vocalLength = (timemark2 - timemark1) / Fs              # Vokallänge mit Pause in Sekunden
            offset1 = (timemark1 / Fs + cutAudio * vocalLength)     # in Sekunden, start des Vokals in Sekunden in wav-file
            dauer = vocalLength * lengthAudio                       # in Sekunden, % vorne und hinten abschneiden um Pause abzutrennen

            y, _ = librosa.load(path=pathAudio, sr=sampleRate, mono=True, offset=offset1,
                                duration=dauer)  # , dtype=<class 'numpy.float32'>, res_type='kaiser_best')
            y = scaler.fit_transform(y.reshape(-1, 1))

            audio.append(y)
            vocalInfo.append(vokal[i])

    audioVocalOne = []
    vocalInfoOne = []
    for i in range(len(audio)):
        if vocalInfo[i]=='a' or vocalInfo[i]=='u':
            audioVocalOne.append(audio[i])
            vocalInfoOne.append(vocalInfo[i])
    #return audio, vocalInfo, sampleRate
    return audioVocalOne, vocalInfoOne, sampleRate