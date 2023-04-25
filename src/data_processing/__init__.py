import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from data_processing.synthetic_data_generator import SineWaveGenerator
from data_processing.data_handler import DataHandler

F_SAMPLING = 48000
SIZE = 512

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

def data_generation():
    datahandler = DataHandler(samplerate = 44100)
    datahandler.load_csv()
    datahandler.create_data(size = 44100*3, new_samplerate = 16000)
