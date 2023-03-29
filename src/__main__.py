import threading
import time
from queue import Queue

import librosa
import matplotlib.pyplot as plt
import numpy as np

from audio_collection import record_audio
from preprocessing_mfsc import preprocessing_mfsc

BLOCKSIZE = 2000
F_SAMPLING = 16000

def main():
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

    print(melspecs.qsize())
    print(len(melspecs.get()[0]))

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(melspecs.get()[0], ref=np.max), sr=F_SAMPLING, x_axis='time', y_axis='mel')
    plt.title('Mel spectrogram')
    plt.show()

if __name__ == '__main__':
    main()
