import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from evaluation.confusion_matrix import calculate_confusion_matrix


def precision_recall_plot(labels, preds, N=200, is_webrtc=False):
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

    #recalls1, precisions1 = precision_recall_curve(original_y, original_y_)
    # recalls2, precisions2 = precision_recall_curve(v2_y, v2_y_)

    # plt.plot(recalls1, precisions1)
    # plt.plot(recalls2, precisions2)
    # #plt.xlim(0, 1.05)
    # #plt.ylim(0, 1.05)
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.show()

    return recalls, precisions


def auc_roc_plot(labels_list, preds_list, model_names, N=200):
    plt.figure(figsize=(8,8))
    thresholds = np.linspace(0, 1, N)

    for i, (labels, preds) in enumerate(zip(labels_list, preds_list)):
        tpr = []
        fpr = []
        for threshold in thresholds:
            cm = calculate_confusion_matrix(labels, preds, threshold)
            tpr.append(recall(cm))
            fpr.append(fp_rate(cm))

        auc_value = roc_auc_score(labels, preds)

        plt.plot(fpr, tpr, label=f'{model_names[i]} | AUC = {auc_value:.4f}')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("ROC Curves", fontsize=16)
    plt.xlabel("FPR", fontsize=12)
    plt.ylabel("TPR", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.savefig('images/roc_curves.png', dpi=300)


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
