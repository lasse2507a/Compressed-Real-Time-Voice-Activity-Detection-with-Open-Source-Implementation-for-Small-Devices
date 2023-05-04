import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from evaluation.data_loader import DataLoader
from evaluation.confusion_matrix import confusion_matrix
from evaluation.metrics import *


def predictions():
    data_loader = DataLoader("data\\output\\prediction_audio_clip_2")
    data, labels = data_loader.load_data_parallel()

    model = tf.keras.models.load_model("models\\cnn_model_original_25(12,8,5).h5")

    predictions = model.predict(x=data, verbose=1)

    return labels, predictions


def precision_recall_plot(labels, predictions, N=100):
    thresholds = np.linspace(0, 1, N)
    precisions = []
    recalls = []

    for threshold in thresholds:
        cm = confusion_matrix(labels, predictions, threshold)
        precisions.append(precision(cm))
        recalls.append(recall(cm))

    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()


def auc_roc(labels, predictions, N=100):
    thresholds = np.linspace(0, 1, N)
    tpr = []
    fpr = []

    for threshold in thresholds:
        cm = confusion_matrix(labels, predictions, threshold)
        tpr.append(recall(cm))
        fpr.append(fp_rate(cm))

    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    auc_value = roc_auc_score(labels, predictions)
    plt.title(f"ROC Curve (AUC = {auc_value:.4f})")
    plt.show()
