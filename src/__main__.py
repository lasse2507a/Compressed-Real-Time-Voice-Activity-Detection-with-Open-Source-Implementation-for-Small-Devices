import threading
import time
from queue import Queue
from audio_collection import record_audio

BLOCKSIZE = 200
F_SAMPLING = 16000

def main():
    recordings = []
    thread_stop_event = threading.Event()
    thread_record_audio = threading.Thread(target=record_audio, daemon=True, args=(F_SAMPLING, BLOCKSIZE, recordings, thread_stop_event))

    thread_record_audio.start()

    time.sleep(10)
    thread_stop_event.set()
    thread_record_audio.join()
    print(len(recordings))
    print(len(recordings[0][0]))

if __name__ == '__main__':
    main()
