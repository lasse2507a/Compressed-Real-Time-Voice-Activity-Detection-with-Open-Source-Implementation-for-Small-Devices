import tensorflow as tf
import visualkeras
from keras.utils import plot_model
from training.cnn_model import CNNModel
from training.data_generator import DataGenerator

def execute_training(training_data_path, validation_data_path):
    model = CNNModel(K=40, L=20, M=10, N=100, keep_prob=0.75)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.SparseCategoricalAccuracy()])

    batch_size = 5000
    training_data = DataGenerator(training_data_path, batch_size)
    validation_data = DataGenerator(validation_data_path, batch_size)

    model.fit(x=training_data,
              validation_data=validation_data,
              epochs=12,
              verbose=1,
              callbacks=tf.keras.callbacks.ModelCheckpoint('models\\epoch_{epoch}'),
              use_multiprocessing=True)

def visualize_model():
    model = CNNModel(K=40, L=20, M=10, N=100, keep_prob=0.75)
    visualkeras.layered_view(model, to_file='model_architecture.png').show()
