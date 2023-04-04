from queue import Queue
import threading
import librosa
import numpy as np

class MFSCPreprocessor:
    def __init__(self, samplerate, blocksize):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.first_round = True
        self.previous_half = Queue()
        self.thread_stop_event = threading.Event()
        self.melspecs = Queue()
        self.melspecs.maxsize = 64

    def start_preprocessing(self, recordings):
        print("MFSC preprocessing started")
        while not self.thread_stop_event.is_set():
            recording = recordings.get()
            recording_copy = np.copy(recording)
            self.previous_half.put(recording_copy[int(self.blocksize/2):])
            if not self.first_round:
                overlapped_recording = np.concatenate((self.previous_half.get(), recording))
                self.melspecs.put(librosa.feature.melspectrogram(y=overlapped_recording,
                                                    sr=self.samplerate,
                                                    n_fft=int(self.blocksize*1.5),
                                                    hop_length=int(self.blocksize*1.5/2),
                                                    window='hann',
                                                    center=True,
                                                    power=2.0,
                                                    n_mels=64,
                                                    fmin=300,
                                                    fmax=8000))
            self.first_round = False
            if self.melspecs.full():
                print("MFSC preprocessing queue full")

    def stop_preprocessing(self):
        self.thread_stop_event.set()
        print("MFSC preprocessing stopped")

'''
if not self.first_round:
                overlapped_recording = np.concatenate((self.previous_half.get(), recording))
                self.melspecs.put(librosa.feature.melspectrogram(y=overlapped_recording,
                                                    sr=self.samplerate,
                                                    n_fft=int(self.blocksize*1.5),
                                                    hop_length=int(self.blocksize*1.5/2),
                                                    window='hann',
                                                    center=True,
                                                    power=2.0,
                                                    n_mels=64,
                                                    fmin=300,
                                                    fmax=8000))
'''