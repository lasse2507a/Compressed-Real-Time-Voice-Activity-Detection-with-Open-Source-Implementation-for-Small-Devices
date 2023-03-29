from queue import Queue

import librosa
import numpy as np

def preprocessing_mfsc(samplerate, blocksize, recordings, melspecs, thread_stop_event):
    print("MFSC preprocessing started")
    previous_half = Queue()
    first_round = True
    while not thread_stop_event.is_set():
        recording = recordings.get()
        print(np.shape(recording))
        recording_copy = np.copy(recording)
        previous_half.put(recording_copy[int(blocksize/2):])
        if not first_round:
            signal = np.concatenate((previous_half.get(), recording))
            print(np.shape(signal))
            melspecs.put(librosa.feature.melspectrogram(y=signal,
                                                        sr=samplerate,
                                                        n_fft=int(blocksize*1.5),
                                                        hop_length=int(blocksize*1.5),
                                                        window='hann',
                                                        center=True,
                                                        power=1.0,
                                                        n_mels=64,
                                                        fmin=300,
                                                        fmax=8000))
        else:
            first_round = False
    print("MFSC preprocessing stopped")
