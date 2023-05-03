import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from evaluation.data_loader import DataLoader
from evaluation.confusion_matrix import confusion_matrix


def evalution():
    data_loader = DataLoader("data/output/prediction_audio_clip_2")
    data, labels = data_loader.load_data_parallel()

    model = tf.keras.models.load_model("models/cnn_model_original_25(12,8,5).h5")

    predictions = model.predict(x=data, verbose=1)

    cm = confusion_matrix(labels, predictions)
    print(cm)
