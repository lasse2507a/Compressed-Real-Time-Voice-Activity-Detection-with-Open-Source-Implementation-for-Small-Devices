from threading import Thread
from queue import Queue

from audio_collection import record_audio

F_SAMPLING = 16000
RECORDING_SIZE = 1000

def main():
    recording_queue = Queue()
    stop = False
    thread_record_audio = Thread(target=record_audio, daemon=True,
                                 args=(RECORDING_SIZE, F_SAMPLING, recording_queue, stop))

    thread_record_audio.start()

    input()
    stop = True

if __name__ == '__main__':
    main()
