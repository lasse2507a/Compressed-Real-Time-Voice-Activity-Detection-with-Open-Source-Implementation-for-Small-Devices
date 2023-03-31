import threading
import time
from queue import Queue

import librosa
import matplotlib.pyplot as plt
import numpy as np

from sine_wave_generator import SineWaveGenerator
from audio_collection import record_audio
from preprocessing_mfsc import preprocessing_mfsc

BLOCKSIZE = 1000
F_SAMPLING = 48000

def real_time_implementation():
    recordings = Queue()
    melspecs = Queue()
    thread_stop_event = threading.Event()
    thread_record_audio = threading.Thread(target=record_audio, daemon=True, args=(F_SAMPLING, BLOCKSIZE, recordings, thread_stop_event))
    thread_preprocessing_mfsc = threading.Thread(target=preprocessing_mfsc, daemon=True, args=(F_SAMPLING, BLOCKSIZE, recordings, melspecs, thread_stop_event))

    thread_record_audio.start()
    thread_preprocessing_mfsc.start()

    time.sleep(3)
    thread_stop_event.set()
    thread_record_audio.join()
    thread_preprocessing_mfsc.join()

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(melspecs.get(), ref=np.max), sr=F_SAMPLING, x_axis='time', y_axis='mel')
    plt.title('Mel spectrogram')
    plt.show()

def frequency_test():
    t = SineWaveGenerator(BLOCKSIZE, F_SAMPLING).time()
    f = SineWaveGenerator(BLOCKSIZE, F_SAMPLING).frequency()
    signal = SineWaveGenerator(BLOCKSIZE, F_SAMPLING).five_sine_wave()
    signal_fft = np.fft.fft(signal)

    plt.figure(figsize=(16,9))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.subplot(2, 1, 2)
    plt.plot(f, abs(signal_fft)[:int(BLOCKSIZE/2)])
    plt.show()

if __name__ == '__main__':
    frequency_test()
    #real_time_implementation()
