from queue import Queue

import librosa
import numpy as np

def preprocessing_mfsc(samplerate, blocksize, recordings, melspecs, thread_stop_event):
    print("MFSC preprocessing started")
    previous_half = Queue()
    first_round = True
    while not thread_stop_event.is_set():
        recording = recordings.get()
        recording_copy = np.copy(recording)
        previous_half.put(recording_copy[int(blocksize/2):])
        if not first_round:
            signal = np.concatenate((previous_half.get(), recording))
            melspecs.put(librosa.feature.melspectrogram(y=signal,
                                                        sr=samplerate,
                                                        n_fft=int(blocksize*1.5),
                                                        hop_length=int(blocksize*1.5/2),
                                                        window='hann',
                                                        center=True,
                                                        power=2.0,
                                                        n_mels=64,
                                                        fmin=300,
                                                        fmax=8000))
        else:
            first_round = False
    print("MFSC preprocessing stopped")

    #plt.figure(figsize=(10, 4))
    #librosa.display.specshow(librosa.power_to_db(melspecs.get(), ref=np.max), sr=F_SAMPLING, x_axis='time', y_axis='mel')
    #plt.title('Mel spectrogram')
    #plt.show()