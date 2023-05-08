from queue import Queue
import threading
import numpy as np
import librosa


class RealTimeMFSCPreprocessor:
    def __init__(self, samplerate, size):
        self.thread_stop_event = threading.Event()
        self.samplerate = samplerate
        self.size = size*2
        self.FFT_size = int(2**np.ceil(np.log2(self.size)))
        self.window = np.hanning(self.size)
        self.frames_MFSC = Queue(maxsize=40)
        self.images_MFSC = Queue()
        self.mel_size = 40


    def start_preprocessing(self, recordings):
        print("MFSC preprocessing started")
        recording = recordings.get()
        previous_recording = recording
        while not self.thread_stop_event.is_set():
            recording = recordings.get()
            frame = np.concatenate((previous_recording, recording))
            previous_recording = recording
            frame = frame * self.window
            zero_padded_frame = np.zeros(self.FFT_size)
            zero_padded_frame[:self.size] = frame
            melspectrogram = librosa.feature.melspectrogram(y=zero_padded_frame, sr=self.samplerate, window='boxcar', n_fft=self.FFT_size,
                                                                    center=False, n_mels=self.mel_size, fmin=300, fmax=8000)
            self.frames_MFSC.put(librosa.power_to_db(melspectrogram, ref=np.max))
            if self.frames_MFSC.full():
                image_MFSC = np.concatenate(list(self.frames_MFSC.queue), axis=1)
                self.images_MFSC.put(image_MFSC)
                for _ in range(5):
                    self.frames_MFSC.get()


    def stop_preprocessing(self):
        self.thread_stop_event.set()
        print("MFSC preprocessing stopped")
