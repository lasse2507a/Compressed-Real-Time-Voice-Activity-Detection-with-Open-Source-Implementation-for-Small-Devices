import numpy as np
from evaluation.confusion_matrix import calculate_confusion_matrix


def precision_recall_curve(labels, preds, N=200, is_webrtc=False):
    precisions = []
    recalls = []

    if is_webrtc:
        for i in range(4):
            cm = calculate_confusion_matrix(labels, preds[i])
            precisions.append(precision(cm))
            recalls.append(recall(cm))
    else:
        thresholds = np.linspace(0, 1, N, endpoint=False)
        for threshold in thresholds:
            cm = calculate_confusion_matrix(labels, preds, threshold)
            precisions.append(precision(cm))
            recalls.append(recall(cm))

    return recalls, precisions


def auc_roc(labels, preds, N=500):
    thresholds = np.linspace(0, 1, N)
    tpr = []
    fpr = []

    for threshold in thresholds:
        cm = calculate_confusion_matrix(labels, preds, threshold)
        tpr.append(recall(cm))
        fpr.append(fp_rate(cm))

    return fpr, tpr


def accuracy(confusion_matrix):
    tp = confusion_matrix['true_positives']
    tn = confusion_matrix['true_negatives']
    fp = confusion_matrix['false_positives']
    fn = confusion_matrix['false_negatives']
    return (tp + tn)/(tp + tn + fp + fn)


def precision(confusion_matrix):
    tp = confusion_matrix['true_positives']
    fp = confusion_matrix['false_positives']
    return tp / (tp + fp)


def recall(confusion_matrix):
    tp = confusion_matrix['true_positives']
    fn = confusion_matrix['false_negatives']
    return tp / (tp + fn)


def f1_score(confusion_matrix):
    tp = confusion_matrix['true_positives']
    fp = confusion_matrix['false_positives']
    fn = confusion_matrix['false_negatives']
    prec = tp / (tp +fp)
    rec = tp / (tp + fn)
    return (2 * prec * rec) / (prec + rec)


def fp_rate(confusion_matrix):
    tn = confusion_matrix['true_negatives']
    fp = confusion_matrix['false_positives']
    return fp / (fp + tn)
