import threading
import time
from queue import Queue
from audio_collection import record_audio
from preprocessing_mfsc import preprocessing_mfsc

BLOCKSIZE = 200
F_SAMPLING = 16000

def main():
    recordings = Queue()
    melspecs = Queue()
    thread_stop_event = threading.Event()
    thread_record_audio = threading.Thread(target=record_audio, daemon=True, args=(F_SAMPLING, BLOCKSIZE, recordings, thread_stop_event))
    thread_preprocessing_mfsc = threading.Thread(target=preprocessing_mfsc, daemon=True, args=(F_SAMPLING, BLOCKSIZE, recordings, melspecs, thread_stop_event))

    thread_record_audio.start()
    thread_preprocessing_mfsc.start()

    time.sleep(5)
    thread_stop_event.set()
    thread_record_audio.join()

    print(recordings.qsize())
    print(len(recordings.get()[0]))

if __name__ == '__main__':
    main()
