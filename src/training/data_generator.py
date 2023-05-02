import os
import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.files = [os.path.join(self.path, file) for file in os.listdir(self.path) if file.endswith(".npy")]
        self.num_files = len(self.files)

    def __len__(self):
        return int(np.floor(self.num_files / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_files = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = []
        batch_labels = []
        for file in batch_files:
            file_data = np.load(file)
            batch_data.append(np.reshape(file_data, (1, 40, 40, 1)))
            label = int(os.path.basename(file).split("_")[-2])
            batch_labels.append(label)
        batch_data = np.concatenate(batch_data, axis=0)
        batch_labels = np.array(batch_labels)
        return batch_data, batch_labels
