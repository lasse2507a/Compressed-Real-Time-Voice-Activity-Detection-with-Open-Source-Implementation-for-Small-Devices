import multiprocessing as mp
import sounddevice as sd
import numpy as np


class AudioRecorder:
    def __init__(self, samplerate, size):
        self.samplerate = samplerate
        self.size = size
        self.recordings = mp.Queue(maxsize=20)
        self.stop_event = mp.Event()
        self.input_stream = sd.InputStream(samplerate=self.samplerate, blocksize=self.size, channels=1, dtype=np.int16)


    def start_recording(self):
        print("audio recording started")
        self.input_stream.start()
        while not self.stop_event.is_set():
            self.recordings.put(np.reshape(self.input_stream.read(frames=self.size)[0], self.size))


    def stop_recording(self):
        self.stop_event.set()
        self.input_stream.abort()
        print("audio recording stopped")
