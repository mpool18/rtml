import glob
import librosa
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np

fsM = 44100
Wn = [2*800/fsM, 2*12800/fsM]
N = 10
b, a = butter(N, Wn, 'band')
for filename in glob.glob("C:/Users/Matthew/Documents/Matthew's Files/College Course Work/Senior Year/Senior Design Project/Bird Songs/*.mp3"):
    data, fs = librosa.load(filename, fsM, True)
    vocal = filtfilt(b, a, data)
    intervals = librosa.effects.split(vocal, top_db=10, frame_length=1024, hop_length=512)
    sounds = np.array([])
    for i in range(np.ma.size(intervals, 0)):
        sounds1 = vocal[intervals[i, 0]:intervals[i, 1]:1]
        sounds = np.concatenate((sounds, sounds1))
    plt.plot(vocal)
    plt.plot(sounds)
    sd.play(sounds, fsM)
    plt.show()
