import librosa
import numpy as np

def preprocessing_mfsc(samplerate, blocksize, recordings, melspecs, thread_stop_event):
    print("MFSC preprocessing started")
    while not thread_stop_event.is_set():
        melspecs.put(librosa.feature.melspectrogram(y=np.transpose(recordings.get()),
                                                    sr=samplerate,
                                                    n_fft=blocksize,
                                                    hop_length=blocksize,
                                                    window='hann',
                                                    center=True,
                                                    power=2.0,
                                                    n_mels=64,
                                                    fmin=300,
                                                    fmax=8000))
    print("MFSC preprocessing stopped")
