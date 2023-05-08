import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def precision_recall_plot(labels_list, preds_list, model_names):
    plt.figure(figsize=(8, 8))

    for i, (labels, preds) in enumerate(zip(labels_list, preds_list)):
        if model_names[i] == 'WebRTC VAD':
            precisions = []
            recalls = []
            for i in range(4):
                cm = calculate_confusion_matrix(labels, preds[i])
                precisions.append(precision(cm))
                recalls.append(recall(cm))
        else:
            precisions, recalls, _ = precision_recall_curve(labels, preds)
            pr_ap = auc(recalls, precisions)

        plt.plot(recalls, precisions, label=f'{model_names[i]} (AP = {pr_ap:.4f})', linewidth=2)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('images/precision_recall_curves.png', dpi=300)


def auc_roc_plot(labels_list, preds_list, model_names):
    plt.figure(figsize=(8,8))

    for i, (labels, preds) in enumerate(zip(labels_list, preds_list)):
        if model_names[i] == 'WebRTC VAD':
            tpr = []
            fpr = []
            for i in range(4):
                cm = calculate_confusion_matrix(labels, preds[i])
                tpr.append(recall(cm))
                fpr.append(fp_rate(cm))
        else:
            fpr, tpr, _ = roc_curve(labels, preds)
            auc_value = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'{model_names[i]} (AUC = {auc_value:.4f})', linewidth=2)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("ROC Curves")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('images/roc_curves.png', dpi=300)


def calculate_confusion_matrix(labels, predictions, threshold=0.5):
    labels = np.array(labels, dtype=np.float32)
    predictions = np.array(predictions, dtype=np.float32)

    for i, prediction in enumerate(predictions):
        if prediction >= threshold:
            predictions[i] = 1
        else:
            predictions[i] = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i, prediction in enumerate(predictions):
        if prediction == 1 and labels[i] == 1:
            tp += 1
        elif prediction == 0 and labels[i] == 0:
            tn += 1
        elif prediction == 1 and labels[i] == 0:
            fp += 1
        elif prediction == 0 and labels[i] == 1:
            fn += 1

    return {'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn}


def accuracy(confusion_matrix):
    tp = confusion_matrix['true_positives']
    tn = confusion_matrix['true_negatives']
    fp = confusion_matrix['false_positives']
    fn = confusion_matrix['false_negatives']
    return (tp + tn)/(tp + tn + fp + fn)


def precision(confusion_matrix):
    tp = confusion_matrix['true_positives']
    fp = confusion_matrix['false_positives']
    if tp + fp == 0:
        return 1
    else:
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
