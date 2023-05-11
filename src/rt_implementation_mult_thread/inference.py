import threading
import numpy as np
import tensorflow as tf


class RealTimeInference:
    def __init__(self, model_name):
        self.model_name = model_name
        self.thread_stop_event = threading.Event()
        self.model = tf.keras.models.load_model(f'models/{model_name}')


    def start_inference(self, images, preds):
        print(f'Inference with {self.model_name} started')
        while not self.thread_stop_event.is_set():
            image = np.expand_dims(np.expand_dims(images.get(), axis=-1), axis=0)
            preds.put(self.model.predict(x=image, verbose=0).ravel())


    def stop_inference(self):
        self.thread_stop_event.set()
        print(f'Inference with {self.model_name} stopped')
