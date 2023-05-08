import threading
from queue import Queue
import numpy as np
import tensorflow as tf


class RealTimeInferenceLite:
    def __init__(self, model_name):
        self.thread_stop_event = threading.Event()
        self.preds = Queue()
        self.model = tf.keras.models.load_model(f'models/{model_name}')
        self.interpreter = tf.lite.Interpreter(model_path=f"models/{model_name}")
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]


    def start_inference(self, images_MFSC):
        while not self.thread_stop_event.is_set():
            image = images_MFSC.get()
            self.interpreter.set_tensor(self.input_details['index'], image.astype(np.float32))
            image = np.expand_dims(np.expand_dims(image, axis=-1), axis=0)
            self.interpreter.allocate_tensors()
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(self.output_details['index']).ravel()
            if prediction >= 0.5:
                print('SPEECH')
            else:
                print('NO-SPEECH')
            self.preds.put(prediction)


    def stop_inference(self):
        self.thread_stop_event.set()
