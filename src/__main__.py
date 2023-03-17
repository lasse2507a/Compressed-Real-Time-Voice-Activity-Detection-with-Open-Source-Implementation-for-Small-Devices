from threading import Thread
from queue import Queue
import time

from audio_collection import record_audio

F_SAMPLING = 44100
RECORDING_SIZE = 4000

def main():
    recording_queue = Queue()
    stop = False
    thread_record_audio = Thread(target=record_audio, daemon=True,
                                 args=(RECORDING_SIZE, F_SAMPLING, recording_queue, stop))

    thread_record_audio.start()

    time.sleep(1)
    stop = True
    print(recording_queue.qsize())

if __name__ == '__main__':
    main()
