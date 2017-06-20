import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

""" ROC """
def plot_roc_curves(roc_data, classes, roc_auc=None):

    sns.set_palette("Set1")
    sns.set_style("ticks", {"axes.facecolor": "1", 
                            'axes.edgecolor': '1',
                            'axes.grid': True,
                            'ytick.color':'0.3'})
    #sns.despine()

    f, ax = plt.subplots(figsize=(9,11))

    cl = range(len(classes))
    
    for c in cl:
        fpr, tpr = roc_data[c]

        label = "{0}".format(classes[c])
        if roc_auc is not None:
            label = "{} (AUC: {:.2f})".format(classes[c], roc_auc[c])
        
        ax.plot(fpr,tpr,label=label, linewidth=1)

    plt.xlabel("False Predictive Ratio (fpr)")
    plt.ylabel("True Predictive Ratio (tpr)")
    plt.title("ROC Curve")
    plt.xlim([-0.01,1.01])
    plt.ylim([-0.01,1.01])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show(f)

""" Confusion Matrix """
def plot_confusion_matrix(confusion_matrix, classes, normalize_axis=None, figsize=(11, 9), text_threshold=20):
    """
    Plots a confusion matrix using seaborn heatmap functionality
    
    @param confusion_matrix: np array [n_classes, n_classes] with rows reference and cols predicted
    @param classes: list of class labels
    @param normalize_axis: 0 sum of rows, 1: sum of cols, None no normalization
    
    @return matplotlib figure
    """
    # Set up the matplotlib figure
    plt.figure()
    f, ax = plt.subplots(figsize=figsize)

    # normalize
    normalized_str = "" # add on at the title
    if normalize_axis is not None:
        
        with np.errstate(divide='ignore'): # ignore divide by zero and replace with 0
            confusion_matrix = np.nan_to_num(confusion_matrix.astype(float) / np.sum(confusion_matrix,axis=normalize_axis))

        if normalize_axis == 0:
            normalized_str=" normalized by sum of references"
        if normalize_axis == 1:
            normalized_str=" normalized by sum of predicted"

    # Draw the heatmap with the mask and correct aspect ratio
    g = sns.heatmap(confusion_matrix,
                square=True,
                linewidths=1, cbar_kws={"shrink": .9}, ax=ax)
    
    
    n_classes = len(classes)
    # if n_classes < threshold plot values in plot
    
    cols = np.arange(0,n_classes)
    rows = np.arange(n_classes-1,-1,-1)
    
    if n_classes < text_threshold:
        for i in np.arange(0,n_classes):
            for j in np.arange(0,n_classes):
                ax.text(cols[i]+.5, rows[j]+.5, "{:0.2f}".format(confusion_matrix[i,j]), 
                         fontdict=None, withdash=False, ha="center", va="center",
                        path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")])


    g.set_title("Confusion Matrix"+normalized_str)
    g.set_xticklabels(classes, rotation=90)
    g.set_yticklabels(classes[::-1], rotation=0)
    g.set_xlabel("predicted")
    g.set_ylabel("reference")
    
    return f, g


""" Confusion Matrix """



""" Barplot interaction """
def stack_data(df):
    df.columns.name = 'measure'
    stacked_df = df.set_index("class").stack()
    stacked_df.name="val"
    stacked_df = pd.DataFrame(stacked_df).reset_index()
    return stacked_df

def barplot(data):
	palette = sns.color_palette("Set3", 8)
	f, ax = plt.subplots(figsize=(11,9))
	sns.despine(left=True)
	ax = sns.barplot(x="class", y="val", hue="measure", data = data, palette=palette, edgecolor="#ffffff", linewidth=2)
	ax.set_title("accuracy metrics")
	ax.set_ylabel("")
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.show()

def accumulate_features(dataloader, limit=200, verbose=False):
    """
    This function queries features as long as an equal amount of <limit> features are available for each class

    Processing in two steps:
        (1) query dataloader.next_batch() as long as at least <limit> number of features are available for each class
        (2) drop all features which are more than <limit> times available

    results in equal number of features per class
    """
    acc_x = None
    acc_y = []
    acc_seq = []

    classes = dataloader.classes

    min_features = 0

    # (1) query as long as with the least features has at least <limit> occurences
    while min_features < limit:
        x, y, seq_lengths = dataloader.next_batch_packed()
        x, y = unroll(x, y, seq_lengths)

        # create histogram of accumulated y
        hist, _ = np.histogram(acc_y, len(classes))

        min_features = hist[np.argmin(hist, axis=0)]

        if verbose:
            print "query data, at least {} features accumulated for each class".format(min_features)

        if acc_x is None:
            acc_x = x
        else:
            acc_x = np.append(x, acc_x, axis=0)
        acc_y = np.append(np.argmax(y, axis=1), acc_y)
        acc_seq = np.append(seq_lengths, acc_seq)

    # (2) drop all features which are more than <limit> times available

    keep_idx = []
    for c in range(0, len(classes)):
        idx = np.where(acc_y == c)[0]
        # keep the first <limit> occurances of class c
        keep_idx = np.append(keep_idx, idx[:limit])

    new_x = acc_x[keep_idx.astype(int), :]
    new_y = acc_y[keep_idx.astype(int)]

    return new_x, new_y

def unroll(x, y, seq_lengths):
    """
    (1) reshapes x and y from 3D -> 2D
        x: [batch x observation x n_input] -> [batch * observations x n_input]
        y: [batch x observation x n_classes] -> [batch * observations x n_classes]

    (2) masks out all features which are later than seq_lengths
    These observations are padded with zeros, and are masked out in this step
    """
    # Reshapes and masks input and output data
    batch_size, max_seqlengths, n_input = x.shape
    np.arange(0, max_seqlengths) * np.ones((batch_size, max_seqlengths))
    ones = np.ones([batch_size, max_seqlengths])
    mask = np.arange(0, max_seqlengths) * ones < (seq_lengths * ones.T).T
    new_x = x[mask]
    new_y = y[mask]
    return new_x, new_y
