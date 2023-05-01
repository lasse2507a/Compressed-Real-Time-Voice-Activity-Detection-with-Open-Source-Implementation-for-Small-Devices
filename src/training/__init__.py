import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import tensorflow as tf
import visualkeras
from keras.utils import plot_model
from training.cnn_model import CNNModel

def load_data_parallel(path):
    data = []
    labels = []
    num_files = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
        files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".npy")]
        for batch in executor.map(load_data, files, chunksize=1000):
            file_data, label = batch
            data.extend(file_data)
            labels.append(label)
            num_files += 1
            if num_files % 10000 == 0:
                print(f"loaded {num_files} files out of {len(files)}")
    data = np.array(data)
    labels = np.array(labels)
    print("data loaded successfully at path: " + str(path))
    return data, labels

def load_data(file):
    file_data = np.load(file)
    label = int(os.path.basename(file).split("_")[-2])
    return np.reshape(file_data, (1, 40, 40)), label


def execute_training(training_data_path, validation_data_path):
    training_data, training_labels = load_data_parallel(training_data_path)
    validation_data, validation_labels = load_data_parallel(validation_data_path)

    model = CNNModel(K=40, L=20, M=10, N=100, keep_prob=0.75)

    model.compile(optimizer=tf.keras.optimizers.Adam(np.hstack((1e-3*np.ones(6),
                                                                1e-4*np.ones(4),
                                                                1e-5*np.ones(2)))),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=tf.keras.metrics.BinaryAccuracy())

    model.fit(x=training_data,
              y=training_labels,
              validation_data=(validation_data, validation_labels),
              batch_size=25000,
              epochs=12,
              steps_per_epoch=len(training_data) // 25000,
              verbose=2,
              callbacks=tf.keras.callbacks.ModelCheckpoint('models\\model_1'),
              use_multiprocessing=True)

def visualize_model():
    model = CNNModel(K=40, L=20, M=10, N=100, classes=2, div=10, batch_size=25000, keep_prob=0.75, learning_rate=np.hstack((1e-3*np.ones(6),
                                                                                                                            1e-4*np.ones(4),
                                                                                                                            1e-5*np.ones(2))))
    visualkeras.layered_view(model, to_file='model_architecture.png').show()
