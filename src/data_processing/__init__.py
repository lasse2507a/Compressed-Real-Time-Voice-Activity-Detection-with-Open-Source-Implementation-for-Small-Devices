import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from data_processing.synthetic_data_generator import SineWaveGenerator
from data_processing.data_handler import DataHandler
from data_processing.preprocessing import Preprocessor

F_SAMPLING = 16000
SIZE = 48000
WINDOW_SIZE = 16000

def data_generation():
    datahandler = DataHandler(samplerate = F_SAMPLING)
    datahandler.load_csv()
    datahandler.create_data(size = SIZE)

def preprocessing():
    print("Preprocessing started")
    preprocessor = Preprocessor(F_SAMPLING, WINDOW_SIZE)
    preprocessor.process('data\\output\\preprocess_test')

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
