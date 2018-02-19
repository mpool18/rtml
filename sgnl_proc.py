import glob
import librosa
from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

fsM = 44100
Wn = [2*800/fsM, 2*12800/fsM]
N = 10
b, a = butter(N, Wn, 'band')
mfccs = np.array([])
for filename in glob.glob("C:/Users/Matthew/Documents/Matthew's Files/College Course Work/Senior Year/Senior Design Project/Bird Songs/*.mp3"):
    data, fs = librosa.load(filename, sr=fsM, mono=True)
    vocal = filtfilt(b, a, data)
    intervals = librosa.effects.split(vocal, top_db=10, frame_length=1024, hop_length=512)
    sounds = np.array([])
    for i in range(np.ma.size(intervals, 0)):
        sounds1 = vocal[intervals[i, 0]:intervals[i, 1]:1]
        sounds = np.concatenate((sounds, sounds1))
    S1 = librosa.feature.melspectrogram(y=sounds, sr=fsM, n_fft=1024, hop_length=512, power=2.0, n_mels=40, fmin=1000, fmax=12000)
    mfcc1 = librosa.feature.mfcc(S=S1, sr=fsM, n_mfcc=13)
    mfcc2 = np.mean(mfcc1, 1)
    if mfccs.size == 0:
        mfccs = mfcc2
    else:
        mfccs = np.vstack((mfccs, mfcc2))
    # sd.play(sounds)
    # plt.plot(vocal)
    # plt.plot(sounds)
    # plt.show()
