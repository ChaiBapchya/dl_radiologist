import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):

    # TODO: Make plots for loss curves and accuracy curves.
    # TODO: You do not have to return the plots.
    # TODO: You can save plots as files by codes here or an interactive way according to your preference.
    # print(train_losses)
    plt.plot(train_losses, label='Training')
    plt.plot(valid_losses, label='Validation')
    plt.legend(loc="best")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.show()
    plt.savefig("train_vs_valid_loss.png")
    plt.close()
    pass
    # print(train_accuracies)
    plt.plot(train_accuracies, label='Training')
    plt.plot(valid_accuracies, label='Validation')
    plt.legend(loc="best")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig("train_vs_valid_accuracy.png")
    plt.close()
    # plt.show()



def plot_confusion_matrix(results, class_names):
    # print(results)
    true_lab, pred_lab = zip(*results)
    cm = confusion_matrix(true_lab, pred_lab)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print(cm)
    # TODO: Make a confusion matrix plot.
    # TODO: You do not have to return the plots.
    # TODO: You can save plots as files by codes here or an interactive way according to your preference.
    pass
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=class_names, yticklabels=class_names,
           title="Confusion Matrix",
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="center")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig("confusion_matrix.png")
    # plt.show()
