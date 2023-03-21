from queue import Queue
import threading
import time
import random
from src import audio_collection

'''

'''

def test_queue_size_of_record_audio():
    random_size = random.randint(1000, 2000)
    random_fs= random.randint(8000, 16000)
    test_queue = Queue()
    thread_stop_event = threading.Event()

    test_thread = threading.Thread(target=audio_collection.record_audio, daemon=True,
                                 args=(random_fs, random_size, test_queue, thread_stop_event))
    test_thread.start()
    time.sleep(1)
    thread_stop_event.set()
    test_thread.join()

    test_queue_size = 0
    while not test_queue.empty():
        test_queue.get()
        test_queue_size = test_queue_size + 1
    assert test_queue_size in range(int(random_fs/random_size - 2), int(random_fs/random_size + 2))

def test_blocksize_of_record_audio():
    random_size = random.randint(1000, 2000)
    random_fs= random.randint(8000, 16000)
    test_queue = Queue()
    thread_stop_event = threading.Event()

    test_thread = threading.Thread(target=audio_collection.record_audio, daemon=True,
                                 args=(random_fs, random_size, test_queue, thread_stop_event))
    test_thread.start()
    time.sleep(1)
    thread_stop_event.set()
    test_thread.join()

    while not test_queue.empty():
        assert len(test_queue.get()[0]) == random_size
