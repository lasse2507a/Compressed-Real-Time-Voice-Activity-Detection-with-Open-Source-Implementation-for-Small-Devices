import threading
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp
from sine_wave_generator import SineWaveGenerator
from data_handler import DataHandler
from real_time_implementation.audio_recorder import AudioRecorder
from real_time_implementation.real_time_preprocessing import RealTimeMFSCPreprocessor

F_SAMPLING = 48000
BLOCKSIZE = 512

def frequency_test():
    t = SineWaveGenerator(BLOCKSIZE, F_SAMPLING).time()
    f = SineWaveGenerator(BLOCKSIZE, F_SAMPLING).frequency()
    signal = SineWaveGenerator(BLOCKSIZE, F_SAMPLING).five_sine_wave(freq4 = 8500)
    signal = np.convolve(signal, sp.firwin(numtaps = 32+1, cutoff = [300, 8000], window = 'hamming', pass_zero = False, fs = F_SAMPLING), mode = "same")
    signal_fft = np.abs(np.fft.fft(signal))[:int(BLOCKSIZE/2)]

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

def real_time_implementation():
    recorder = AudioRecorder(F_SAMPLING, BLOCKSIZE)
    preprocessor = RealTimeMFSCPreprocessor(F_SAMPLING, BLOCKSIZE)

    thread_recorder = threading.Thread(target=recorder.start_recording, daemon=True)
    thread_preprocessor = threading.Thread(target=preprocessor.start_preprocessing, args=(recorder.recordings,), daemon=True)

    thread_recorder.start()
    thread_preprocessor.start()

    time.sleep(1)

    recorder.stop_recording()
    preprocessor.stop_preprocessing()
    thread_recorder.join()
    thread_preprocessor.join()

    f = SineWaveGenerator(int(BLOCKSIZE), F_SAMPLING).frequency()
    signal = preprocessor.melspecs.get()
    plt.figure(figsize=(16,9))
    plt.plot(f, signal, ".-")
    plt.show()

def batch_implementation():
    print("batch implementation")

if __name__ == '__main__':
    #frequency_test()
    data_generation()
    #batch_implementation()
    #real_time_implementation()
