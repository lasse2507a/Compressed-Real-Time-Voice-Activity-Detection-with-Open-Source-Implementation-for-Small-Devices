import os
from concurrent.futures import ThreadPoolExecutor
import librosa
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import webrtcvad
from sklearn.metrics import roc_auc_score
from evaluation.data_loader import DataLoader
from evaluation.confusion_matrix import confusion_matrix
from evaluation.metrics import *


def predictions():
    data_loader_mfsc = DataLoader("data\\output\\prediction_audio_clip_2\\mfsc_400samples")
    data, labels = data_loader_mfsc.load_data_parallel()

    model = tf.keras.models.load_model("models\\cnn_model_original_25(12,8,5).h5")
    preds = model.predict(x=data, verbose=1)

    return labels, preds


def load_file(file):
    file_data, _ = librosa.load(file, sr=16000, mono=True)
    clips = np.frombuffer(file_data, dtype=np.int16).reshape(-1, 2)
    label = int(os.path.basename(file).split("_")[-1][0])
    return clips, label

def predictions_webrtc():
    labels = []
    path = "data/output/prediction_audio_clip_2/audio_clip_2_480samples"
    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".wav")]
    preds = np.empty((4, len(files)))

    batch_size = 100
    num_batches = int(np.floor(len(files) / batch_size))

    with ThreadPoolExecutor() as executor:
        for batch_idx in range(num_batches):
            batch_files = files[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            results = list(executor.map(load_file, batch_files))
            clips, batch_labels = zip(*results)
            labels += batch_labels
            clips = np.array(clips)
            clips = np.reshape(clips, (batch_size, -1))
            print(f"batch {batch_idx+1}/{num_batches}: files loaded for prediction with WebRTC")

            model_webrtc = webrtcvad.Vad()

            for i in range(4):
                model_webrtc.set_mode(i)
                print(f"batch {batch_idx+1}/{num_batches}: WebRTC mode: {i}")
                for j, clip in enumerate(clips):
                    preds[i][batch_idx * batch_size + j] = model_webrtc.is_speech(buf=clip, sample_rate=16000)

    return labels, preds


def precision_recall_plot(labels, preds, N=100, is_webrtc=False):
    thresholds = np.linspace(0, 1, N)
    precisions = []
    recalls = []

    if is_webrtc:
        for i in range(len(preds[0])):
            cm = confusion_matrix(labels, preds[i], 0.5)
            precisions.append(precision(cm))
            recalls.append(recall(cm))
    else:
        for threshold in thresholds:
            cm = confusion_matrix(labels, preds, threshold)
            precisions.append(precision(cm))
            recalls.append(recall(cm))

    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()


def auc_roc(labels, preds, N=100):
    thresholds = np.linspace(0, 1, N)
    tpr = []
    fpr = []

    for threshold in thresholds:
        cm = confusion_matrix(labels, preds, threshold)
        tpr.append(recall(cm))
        fpr.append(fp_rate(cm))

    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    auc_value = roc_auc_score(labels, preds)
    plt.title(f"ROC Curve (AUC = {auc_value:.4f})")
    plt.show()
