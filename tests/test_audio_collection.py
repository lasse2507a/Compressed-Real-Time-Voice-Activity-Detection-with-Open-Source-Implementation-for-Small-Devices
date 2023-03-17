import time
import random
from threading import Thread
from queue import Queue
from src import audio_collection

def test_record_audio():
    random_size = random.randint(1000, 2000)
    random_fs= random.randint(8000, 16000)
    test_queue = Queue()
    stop = False

    thread_record_audio = Thread(target=audio_collection.record_audio, daemon=True,
                                 args=(random_size,
                                       random_fs,
                                       test_queue,
                                       stop))
    thread_record_audio.start()
    time.sleep(1)
    stop = True

    assert test_queue.qsize() >= random_fs/random_size or test_queue.qsize() <= random_fs/random_size + 1
