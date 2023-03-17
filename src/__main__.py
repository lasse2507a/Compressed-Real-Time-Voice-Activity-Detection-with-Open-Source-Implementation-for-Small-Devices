import threading
import time

from audio_collection import record_audio

RECORDING_SIZE = 1000
F_SAMPLING = 16000

def main():
    recordings = []
    thread_stop_event = threading.Event()
    thread_record_audio = threading.Thread(target=record_audio, daemon=True,
                                 args=(RECORDING_SIZE, F_SAMPLING, recordings, thread_stop_event))

    thread_record_audio.start()

    time.sleep(5)
    thread_stop_event.set()
    thread_record_audio.join()
    print(len(recordings))
    print(len(recordings[0]))

if __name__ == '__main__':
    main()
