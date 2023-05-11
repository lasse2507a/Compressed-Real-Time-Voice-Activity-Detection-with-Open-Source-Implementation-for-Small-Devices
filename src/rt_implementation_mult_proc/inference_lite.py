import multiprocessing as mp
import numpy as np
import tensorflow as tf


class RealTimeInferenceLite:
    def __init__(self, model_name):
        self.preds = mp.Queue()
        self.stop_event = mp.Event()
        self.interpreter = tf.lite.Interpreter(model_path=f"models\\{model_name}")
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]


    def start_inference(self, images_MFSC):
        print('lite inference started')
        while not self.stop_event.is_set():
            self.interpreter.allocate_tensors()
            image = images_MFSC.get()
            image = np.expand_dims(np.expand_dims(image, axis=-1), axis=0)
            self.interpreter.set_tensor(self.input_details['index'], image.astype(np.float32))
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(self.output_details['index']).ravel()
            #preds.put(prediction)
            print(prediction)


    def stop_inference(self):
        self.stop_event.set()
        print('lite inference stopped')
