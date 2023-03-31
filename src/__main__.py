import threading
import time

import librosa
import matplotlib.pyplot as plt
import numpy as np

from sine_wave_generator import SineWaveGenerator
from audio_recorder import AudioRecorder
#from preprocessing_mfsc import preprocessing_mfsc

F_SAMPLING = 48000
BLOCKSIZE = 128

def real_time_implementation():
    audio_recorder = AudioRecorder(F_SAMPLING, BLOCKSIZE)
    thread_audio_recorder = threading.Thread(target=audio_recorder.start_recording, daemon=True)
    thread_audio_recorder.start()

    time.sleep(1)

    audio_recorder.stop_recording()
    thread_audio_recorder.join()

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
    #frequency_test()
    real_time_implementation()
