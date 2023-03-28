import librosa

def preprocessing_mfsc(samplerate, blocksize, recordings, melspecs, thread_stop_event):
    print("MFSC preprocessing started")
    while not thread_stop_event.is_set():
        melspecs.put(librosa.feature.melspectrogram(y=recordings.get(),
                                                    sr=samplerate,
                                                    S=None,
                                                    n_fft=blocksize,
                                                    hop_length=blocksize,
                                                    window='hann',
                                                    center=False,
                                                    power=1.0))
    print("MFSC preprocessing stopped")
