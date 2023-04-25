import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from data_processing.synthetic_data_generator import SineWaveGenerator
from data_processing.data_handler import DataHandler

F_SAMPLING = 16000
SIZE = 16000*3

def data_generation():
    datahandler = DataHandler(samplerate = F_SAMPLING)
    datahandler.load_csv()
    datahandler.create_data(size = SIZE)

def frequency_test():
    t = SineWaveGenerator(SIZE, F_SAMPLING).time()
    f = SineWaveGenerator(SIZE, F_SAMPLING).frequency()
    signal = SineWaveGenerator(SIZE, F_SAMPLING).five_sine_wave(freq4 = 8500)
    signal = np.convolve(signal, sp.firwin(numtaps = 32+1, cutoff = [300, 8000], window = 'hamming', pass_zero = False, fs = F_SAMPLING), mode = "same")
    signal_fft = np.abs(np.fft.fft(signal))[:int(SIZE/2)]

    plt.figure(figsize=(16,9))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, ".-")
    plt.subplot(2, 1, 2)
    plt.plot(f, signal_fft, ".-")
    plt.show()
