import threading
import time
import matplotlib.pyplot as plt
import numpy as np
from sine_wave_generator import SineWaveGenerator
from audio_recorder import AudioRecorder
from preprocessing_mfsc import MFSCPreprocessor

F_SAMPLING = 48000
BLOCKSIZE = 128

def real_time_implementation():
    recorder = AudioRecorder(F_SAMPLING, BLOCKSIZE)
    preprocessor = MFSCPreprocessor(F_SAMPLING, BLOCKSIZE)

    thread_recorder = threading.Thread(target=recorder.start_recording, daemon=True)
    thread_preprocessor = threading.Thread(target=preprocessor.start_preprocessing, args=(recorder.recordings,), daemon=True)

    thread_recorder.start()
    thread_preprocessor.start()

    time.sleep(1)

    recorder.stop_recording()
    preprocessor.stop_preprocessing()
    thread_recorder.join()
    thread_preprocessor.join()

    #plt.figure(figsize=(10, 4))
    #librosa.display.specshow(librosa.power_to_db(melspecs.get(), ref=np.max), sr=F_SAMPLING, x_axis='time', y_axis='mel')
    #plt.title('Mel spectrogram')
    #plt.show()

def frequency_test():
    t = SineWaveGenerator(BLOCKSIZE, F_SAMPLING).time()
    f = SineWaveGenerator(BLOCKSIZE, F_SAMPLING).frequency()
    signal = SineWaveGenerator(BLOCKSIZE, F_SAMPLING).five_sine_wave()
    signal_fft = np.abs(np.fft.fft(signal))[:int(BLOCKSIZE/2)]

    plt.figure(figsize=(16,9))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, ".-")
    plt.subplot(2, 1, 2)
    plt.plot(f, signal_fft, ".-")
    plt.show()

if __name__ == '__main__':
    frequency_test()
    #real_time_implementation()
