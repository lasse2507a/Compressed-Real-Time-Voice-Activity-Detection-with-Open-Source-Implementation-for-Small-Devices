from threading import Thread
from queue import Queue
import time
import random
import sounddevice as sd
from src import audio_collection

def test_queue_size_of_record_audio():
    random_size = random.randint(1000, 2000)
    random_fs= random.randint(8000, 16000)
    test_queue = Queue()
    stop = False

    test_thread1 = Thread(target=audio_collection.record_audio, daemon=True,
                                 args=(random_size, random_fs, test_queue, stop))
    test_thread1.start()
    time.sleep(1)
    stop = True
    sd.stop()

    assert test_queue.qsize()*random_size >= random_fs and test_queue.qsize()*random_size <= random_fs + random_size


def test_recording_size_of_record_audio():
    random_size = random.randint(1000, 2000)
    random_fs= random.randint(8000, 16000)
    test_queue = Queue()
    test_queue.maxsize = 200
    stop = False

    test_thread2 = Thread(target=audio_collection.record_audio, daemon=True,
                                 args=(random_size, random_fs, test_queue, stop))
    test_thread2.start()
    time.sleep(1)
    stop = True

    assert len(test_queue.get()) == random_size
