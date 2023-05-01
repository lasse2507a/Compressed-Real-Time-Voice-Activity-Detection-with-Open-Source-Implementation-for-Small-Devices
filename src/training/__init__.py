import os
import numpy as np
import tensorflow as tf
import visualkeras
from keras.utils import plot_model
from training.cnn_model import CNNModel
from training.data_generator import DataGenerator

def execute_training(training_data_path, validation_data_path):
    model = CNNModel(K=40, L=20, M=10, N=100, keep_prob=0.75)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), # Not same learning rate as paper
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=tf.keras.metrics.BinaryAccuracy())

    batch_size = 5000
    training_data = DataGenerator(training_data_path, batch_size)
    validation_data = DataGenerator(validation_data_path, batch_size)

    model.fit(x=training_data,
              validation_data=validation_data,
              validation_steps=len(validation_data),
              epochs=12,
              verbose=1,
              callbacks=tf.keras.callbacks.ModelCheckpoint('models\\epoch_{epoch}'),
              use_multiprocessing=True)

def visualize_model():
    model = CNNModel(K=40, L=20, M=10, N=100, keep_prob=0.75)
    visualkeras.layered_view(model, to_file='model_architecture.png').show()
