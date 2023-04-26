import os
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from data_processing.mel_energy_filterbank import MelEnergyFilterbank

class Preprocessor:
    def __init__(self, samplerate, size):
        self.samplerate = samplerate
        self.size = size
        self.FFT_size = 2**math.ceil(math.log2(self.size))
        self.window = signal.windows.hann(self.size)
        self.first_round = True
        self.frames = []
        self.frames_FFT = []
        self.frames_MFSC = []
        self.mel_size = 40
        self.mel_frame = np.zeros((self.mel_size, self.FFT_size))
        self.mel_filterbank = MelEnergyFilterbank(self.FFT_size, self.mel_size, self.samplerate)

    def process(self, data_path):
        for file in os.listdir(data_path):
            _, current_file = wavfile.read(os.path.join(data_path, file))
            current_file = current_file[:, 0]
            print(np.shape(current_file))
            for i in range(0, int(len(current_file)-(self.size/2)), int(self.size/2)):
                self.frames.append(current_file[i:i+self.size] * self.window)
                print(str(os.path.join(data_path, file)) + str(i) + " window")

        for j in range(len(self.frames)):
            self.frames_FFT.append(np.abs(np.fft.fft(self.frames[j], self.FFT_size))**2)
            print(str(j) + " FFT")

        print(np.shape(self.mel_frame[:, 0]))
        for k in range(len(self.frames_FFT)):
            for l in range(self.FFT_size):
                self.mel_frame[:, l] = self.mel_filterbank.compute(self.frames_FFT[k])
            self.frames_MFSC.append(self.mel_frame)
            print(str(k) + " MFSC")
            print(self.frames_MFSC[0])

        plt.figure(figsize=(16,9))
        plt.subplot(4, 1, 1)
        plt.plot(np.linspace(0, 16000, 16000), self.window, ".-")
        plt.subplot(4, 1, 2)
        plt.plot(np.linspace(0, 16000, 16000), self.frames[0], ".-")
        plt.subplot(4, 1, 3)
        plt.plot(np.linspace(0, 16384, 16384), self.frames_FFT[0], ".-")
        plt.subplot(4, 1, 4)
        plt.plot(np.linspace(0, 40, 40), self.frames_MFSC[0], ".-")
        plt.show()
