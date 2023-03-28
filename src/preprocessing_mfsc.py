import librosa
import numpy as np

def preprocessing_mfsc(samplerate, blocksize, recordings, melspecs, thread_stop_event):
    print("MFSC preprocessing started")
    while not thread_stop_event.is_set():
        melspecs.put(librosa.feature.melspectrogram(y=recordings.get(), sr=samplerate, n_fft=blocksize))
    print("MFSC preprocessing stopped")
