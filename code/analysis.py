import numpy as np
import pvml
import matplotlib.pyplot as plt


def make_confusion_matrix(predictions, lables):
    cmat = np.zeros((35, 35))
    for i in range(predictions.size):
        cmat[lables[i], predictions[i]] += 1
    return cmat


def show_confusion_matrix(Y, predictions, cakes):
    classes = Y.max() + 1
    cm = np.empty((classes, classes))
    for klass in range(classes):
        sel = (Y == klass).nonzero()
        counts = np.bincount(predictions[sel], minlength=classes)
        cm[klass, :] = 100 * counts / max(1, counts.sum())
    plt.figure(3, figsize=(15, 15))
    plt.clf()
    plt.xticks(range(classes), cakes, rotation=45)
    plt.yticks(range(classes), cakes)
    plt.imshow(cm, vmin=0, vmax=100, cmap=plt.cm.Reds)
    for i in range(classes):
        for j in range(classes):
            txt = "{:.1f}".format(cm[i, j], ha="center", va="center")
            col = ("black" if cm[i, j] < 75 else "white")
            plt.text(j - 0.25, i, txt, color=col)
    plt.title("Confusion matrix")
