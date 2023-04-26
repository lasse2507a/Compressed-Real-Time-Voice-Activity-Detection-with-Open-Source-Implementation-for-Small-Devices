import os
import math
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from data_processing.mel_energy_filterbank import MelEnergyFilterbank

class Preprocessor:
    def __init__(self, samplerate, size):
        self.samplerate = samplerate
        self.size = size
        self.FFT_size = math.ceil(math.log2(size))
        self.window = signal.windows.hann(size)
        self.first_round = True
        self.frames = []
        self.frames_FFT = []
        self.frames_MFSC = []
        self.mel_size = 40
        self.mel_frame = np.zeros((self.mel_size, self.FFT_size))
        self.mel_Filterbank = MelEnergyFilterbank(self.FFT_size, self.mel_size, self.samplerate)

    def process(self, data_path):
        for file in os.listdir(data_path):
            current_file = wavfile.read(os.path.join(data_path, file))[1]
            for i in range(0, len(current_file)-(self.size/2), self.size/2):
                self.frames.append(np.convolve(current_file[i:i+self.size], self.window, 'same'))

        for j in range(len(self.frames)):
            self.frames_FFT.append(np.abs(np.fft.fft(self.frames[j], self.FFT_size))**2)

        for k in range(len(self.frames_FFT)):
            for l in range(self.FFT_size):
                self.mel_frame[:, l] = self.mel_Filterbank.compute(self.frames_FFT[k])
            self.frames_MFSC.append(self.mel_frame)
