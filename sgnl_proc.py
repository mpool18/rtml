import glob
import librosa
from scipy.signal import buttord, butter, lfilter
import matplotlib.pyplot as plt

N, Wn = buttord([2*1000/44100, 2*12000/44100], [2*800/44100, 2*13000/44100], 1, 30)
b, a = butter(N, Wn, 'band')
fsM=44100

for filename in glob.glob("C:/Users/Matthew/Documents/Matthew's Files/College Course Work/Senior Year/Senior Design Project/Bird Songs/*.wav"):
    data, fs = librosa.load(filename, 44100, True)
    print(fs)
    vocal = lfilter(b, a, data)
    print(data)
    print(vocal)
    plt.plot(data)
    plt.plot(vocal)
    plt.show()
