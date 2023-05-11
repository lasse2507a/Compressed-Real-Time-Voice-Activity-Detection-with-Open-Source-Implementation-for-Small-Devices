from queue import Queue
import threading
import sounddevice as sd
import numpy as np


class AudioRecorder:
    def __init__(self, samplerate, size):
        self.samplerate = samplerate
        self.size = size
        self.recordings = Queue() # maxsize could be added to prevent latency of inital loading of the system
        self.thread_stop_event = threading.Event()
        self.input_stream = sd.InputStream(samplerate=self.samplerate, blocksize=self.size, channels=1, dtype=np.int16)


    def start_recording(self):
        print("Audio recording started")
        self.input_stream.start()
        while not self.thread_stop_event.is_set():
            self.recordings.put(np.reshape(self.input_stream.read(frames=self.size)[0], self.size))


    def stop_recording(self):
        self.thread_stop_event.set()
        self.input_stream.abort()
        print("Audio recording stopped")
