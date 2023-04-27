import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import librosa
from data_processing.mel_energy_filterbank import MelEnergyFilterbank

class Preprocessor:
    def __init__(self, samplerate, size):
        self.samplerate = samplerate
        self.size = size
        self.hop_length = int(size/2)
        self.FFT_size = int(2**np.ceil(np.log2(self.size))) #Unused in Librosa implementation
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
            current_file, _ = librosa.load(os.path.join(data_path, file), sr=self.samplerate)
            num_frames = int(np.floor((len(current_file) - self.hop_length) / self.hop_length))
            mfsc_list = []
            for i in range(num_frames):
                start = int(i * self.hop_length)
                end = start + self.size
                frame = current_file[start:end] * np.hanning(self.size)
                zero_padded_frame = np.zeros(self.FFT_size)
                zero_padded_frame[:self.size] = frame
                melspectrogram = librosa.feature.melspectrogram(y=zero_padded_frame, sr=self.samplerate, window='boxcar', n_fft=self.FFT_size, hop_length=self.hop_length,
                                                                center=False, n_mels=self.mel_size, fmin=300, fmax=8000)
                mfsc = librosa.power_to_db(melspectrogram, ref=np.max)
                mfsc_list.append(mfsc)
            mfsc = np.concatenate(mfsc_list, axis=1)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mfsc, x_axis='time', y_axis='mel', cmap='coolwarm')
            plt.colorbar()
            plt.title('MFSC Features')
            plt.tight_layout()
            plt.show()

    def process2(self, data_path):
        for file in os.listdir(data_path):
            _, current_file = wavfile.read(os.path.join(data_path, file))
            current_file = current_file[:, 0]
            print(np.shape(current_file))
            for i in range(0, int(len(current_file)-(self.size/2)), int(self.size/2)):
                self.frames.append(current_file[i:i+self.size] * self.window)
                print(str(os.path.join(data_path, file)) + str(i) + " window")

        for j in range(len(self.frames)):
            self.frames_FFT.append(np.abs(np.fft.fft(self.frames[j], self.FFT_size))**2)
            print(str(j) + " FFT Power Spectrum")

        for k in range(len(self.frames_FFT)):
            for l in range(self.FFT_size):
                self.mel_frame[:, l] = self.mel_filterbank.compute(self.frames_FFT[k])
            self.frames_MFSC.append(self.mel_frame)
            print(str(k) + " MFSC")
            print(self.frames_MFSC[0])
