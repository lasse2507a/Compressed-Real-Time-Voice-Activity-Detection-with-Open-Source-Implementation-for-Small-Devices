def accuracy(confusion_matrix):
    tp = confusion_matrix['true_positives']
    tn = confusion_matrix['true_negatives']
    fp = confusion_matrix['false_positives']
    fn = confusion_matrix['false_negatives']
    return (tp + tn)/(tp + tn + fp + fn)


def precision(confusion_matrix):
    tp = confusion_matrix['true_positives']
    fp = confusion_matrix['false_positives']
    return tp / (tp +fp)


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