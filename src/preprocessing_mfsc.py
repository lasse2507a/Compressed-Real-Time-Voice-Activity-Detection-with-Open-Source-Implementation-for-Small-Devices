from queue import Queue
import threading
import scipy.signal as sp
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
                filtered_overlapped_recording = np.convolve(overlapped_recording, sp.firwin(numtaps = 64+1, cutoff = [300, 8000], window = 'hamming', pass_zero = 'bandpass', fs = self.samplerate), mode = "same")
                fft_overlapped_recording = np.abs(np.fft.fft(filtered_overlapped_recording))[:int(self.blocksize*1.5/2)]
                self.melspecs.put(fft_overlapped_recording)
            self.first_round = False
            if self.melspecs.full():
                print("MFSC preprocessing queue full")
                self.thread_stop_event.set()

    def stop_preprocessing(self):
        self.thread_stop_event.set()
        print("MFSC preprocessing stopped")
