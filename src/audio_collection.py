import sounddevice as sd
import numpy as np
from queue import Queue
import threading

def record_audio(samplerate, blocksize, recordings, thread_stop_event):
    print("Audio recording started")
    input_stream = sd.InputStream(samplerate=samplerate, blocksize=blocksize, channels=1, dtype=np.float32)
    input_stream.start()
    while not thread_stop_event.is_set():
        recordings.put(np.reshape(input_stream.read(frames=blocksize)[0], 256))
    if thread_stop_event.is_set():
        input_stream.abort()
    print("Recording stopped")

class AudioRecorder:
    def __init__(self, samplerate, blocksize):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.recordings = Queue()
        self.thread_stop_event = threading.Event()
        self.input_stream = sd.InputStream(samplerate=self.samplerate, blocksize=self.blocksize, channels=1, dtype=np.float32)

    def start_recording(self):
        print("Audio recording started")
        self.input_stream.start()
        while not self.thread_stop_event.is_set():
            self.recordings.put(np.reshape(self.input_stream.read(frames=self.blocksize)[0], 256))

    def stop_recording(self):
        if self.input_stream.active:
            self.input_stream.stop()
        self.input_stream.close()
        self.thread_stop_event.set()
        print("Recording stopped")
