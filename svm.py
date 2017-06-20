import psycopg2
from DataLoader import DataLoader
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import os
import cPickle as pickle

conn = psycopg2.connect(os.environ["FIELDDBCONNECTSTRING"])

batch_size = 500
limit = 3000

train_dataloader = DataLoader(conn=conn, batch_size=batch_size, sql_where="where is_train=True", debug=False,
                                  do_shuffle=True, tablename="raster_label_fields", packed_table="packed_batches", pack_size=500)
test_dataloader = DataLoader(conn=conn, batch_size=batch_size, sql_where="where is_train=False", debug=False,
                                  do_shuffle=True, tablename="raster_label_fields", packed_table="packed_batches", pack_size=500)

classes = train_dataloader.classes


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

# query 3000 features of each class
x, y = accumulate_features(train_dataloader, limit=3000)

x_test, y_test = accumulate_features(test_dataloader, limit=3000)

y_true = y

clf = SVC(C=1.0, cache_size=7000, class_weight=None, coef0=0.0,
          decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
          max_iter=-1, probability=True, random_state=None, shrinking=True,
          tol=0.001, verbose=False)

c_lim = (-2,7)
g_lim = (-2,4)

param_grid = [
        {'C': [10**exp for exp in range(*c_lim)], 'gamma': [10**exp for exp in range(*g_lim)], 'kernel': ['rbf']},
    ]


pickle.dump( clf, open( "svm/clf.pkl", "wb" ) )

np.save("svm/param_grid.npy",param_grid)

grid_search = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1, cv=10, verbose=True)
grid_search.fit(x, y)

if not os.path.exists("svm"):
    os.mkdir("svm")

pickle.dump( grid_search, open( "svm/grid_search.npy", "wb" ) )

np.save("svm/x.npy",x)
np.save("svm/y.npy",y)
np.save("svm/x_test.npy",x_test)
np.save("svm/y_test.npy",y_test)
np.save("svm/limit.npy",limit)
np.save("svm/classes.npy",classes)
