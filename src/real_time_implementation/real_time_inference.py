import threading
from queue import Queue
import tensorflow as tf


class RealTimeInference:
    def __init__(self, model_name):
        self.thread_stop_event = threading.Event()
        self.model = tf.keras.models.load_model(f'models/{model_name}')
        self.preds = Queue()


    def start_inference(self, images):
        while not self.thread_stop_event.is_set():
            prediction = self.model.predict(images.get())
            self.preds.put(prediction)

    def stop_inference(self):
        self.thread_stop_event.set()
