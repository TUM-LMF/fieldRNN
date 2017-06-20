import matplotlib.pyplot as plt
import numpy as np
import itertools
import io

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, max=20):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    :type max: int
    """

    c = []
    for cl in classes:
        c.append(cl[:max])

    classes = np.array(c)


    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        """
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        """

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    ## png buffer for tensorflow
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200)
    buf.seek(0)
    return buf

def main():
    cnf_matrix = [[33, 2, 0, 0, 0, 0, 0, 0, 0, 1, 3],
                [3, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 4, 41, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 30, 0, 6, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 38, 10, 0, 0, 0, 0, 0],
                [0, 0, 0, 3, 1, 39, 0, 0, 0, 0, 4],
                [0, 2, 2, 0, 4, 1, 31, 0, 0, 0, 2],
                [0, 1, 0, 0, 0, 0, 0, 36, 0, 2, 0],
                [0, 0, 0, 0, 0, 0, 1, 5, 37, 5, 1],
                [3, 0, 0, 0, 0, 0, 0, 0, 0, 39, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38]]

    #cm = np.array([np.array(xi) for xi in cnf_matrix])

    class_names = ["A","B","C","D","E","F","G","I","J","K","L"]

    plt.figure()
    plot_confusion_matrix(np.array(cnf_matrix), classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.show()



if __name__ == '__main__':
    main()