import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from data_processing.synthetic_data_generator import SineWaveGenerator
from data_processing.data_handler import DataHandler
from data_processing.preprocessing import Preprocessor

F_SAMPLING = 16000
SIZE = 17200
WINDOW_SIZE = 400

def data_generation():
    datahandler = DataHandler(samplerate = F_SAMPLING)
    datahandler.load_csv()
    datahandler.create_data(size = SIZE)

def preprocessing():
    preprocessor = Preprocessor(F_SAMPLING, WINDOW_SIZE)
    preprocessor.process('data\\output\\training_clip_len_17200samples')
    preprocessor.process('data\\output\\test_clip_len_17200samples')

def plot_mfsc(data_path='data\\output\\preprocess_test\\mfsc_examples'):
    for i, file in enumerate(os.listdir(data_path)):
        image_MFSC = np.load(os.path.join(data_path, file))
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(image_MFSC, y_coords=librosa.mel_frequencies(n_mels=40, fmin=300, fmax=8000), x_axis='time', y_axis='mel', cmap='coolwarm')
        plt.colorbar()
        plt.title('MFSC Features')
        plt.tight_layout()
        plt.savefig(f'data\\output\\preprocess_test\\images\\{file}_mfsc_example{i+1}.png')

def frequency_test():
    t = SineWaveGenerator(SIZE, F_SAMPLING).time()
    f = SineWaveGenerator(SIZE, F_SAMPLING).frequency()
    signal = SineWaveGenerator(SIZE, F_SAMPLING).five_sine_wave()
    signal_fft = np.abs(np.fft.fft(signal))[:int(SIZE/2)]
    signal_fft2 = np.abs(np.fft.fft(signal))**2

    plt.figure(figsize=(16,9))
    plt.subplot(3, 1, 1)
    plt.plot(t, signal, ".-")
    plt.subplot(3, 1, 2)
    plt.plot(f, signal_fft, ".-")
    plt.subplot(3, 1, 3)
    plt.plot(np.linspace(0, F_SAMPLING, SIZE), signal_fft2, ".-")
    plt.show()
