import numpy as np
import pandas as pd
import cPickle as pickle
import sklearn

def inverse_confusion_matrix(cm):
    # inverse functino of sklearn confusion_matrix
    rows, cols = cm.shape
    y_pred = []
    y_true = []
    for r in range(rows):
        for c in range(cols):
            for i in range(cm[c,r]):
                y_pred.append(r)
                y_true.append(c)
    return y_pred, y_true

def calculate_accuracy_metrics2(y_true, y_pred, labels):
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    sklearn.metrics.precision_score(y_true, y_pred)
    precision, recall, fscore = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, labels=labels)


def calculate_accuracy_metrics(confusion_matrix):
    """
    https: // en.wikipedia.org / wiki / Confusion_matrix

    Calculates over all accuracy and per class classification metrics from confusion matrix

    :param confusion_matrix numpy array [n_classes, n_classes] rows True Classes, columns predicted classes:
    :return overall accuracy
            and per class metrics as list[n_classes]:
    """
    if type(confusion_matrix) == list:
        confusion_matrix = np.array(confusion_matrix)

    confusion_matrix = confusion_matrix.astype(float)

    total = np.sum(confusion_matrix)
    n_classes, _ = confusion_matrix.shape
    classes = []
    overall_accuracy = np.sum(np.diag(confusion_matrix)) / total

    for c in range(n_classes):
        tp = confusion_matrix[c, c]
        fp = np.sum(confusion_matrix[:, c]) - tp
        fn = np.sum(confusion_matrix[c, :]) - tp
        tn = np.sum(np.diag(confusion_matrix)) - tp

        accuracy = (tp + tn) / (tp + fp + fn + tn)

        if (tp + fn) > 0:
            recall = tp / (tp + fn)  # aka sensitivity, hitrate, true positive rate
        else:
            recall = None

        if (fp + tn) > 0:
            specificity = tn / (fp + tn)  # aka true negative rate
        else:
            specificity = None

        if (tp + fp) > 0:
            precision = tp / (tp + fp)  # aka positive predictive value
        else:
            precision = None

        if (2 * tp + fp + fn) > 0:
            fscore = (2 * tp) / (2 * tp + fp + fn)
        else:
            fscore = None

        # http://epiville.ccnmtl.columbia.edu/popup/how_to_calculate_kappa.html
        probability_observed = np.sum(np.diag(confusion_matrix)) / total
        colsum = np.sum(confusion_matrix[:, c])
        rowsum = np.sum(confusion_matrix[c, :])
        probability_expected = (colsum * rowsum / total**2) + (total - colsum) * (total - rowsum) / total**2

        kappa = (probability_observed - probability_expected) / (1 - probability_expected)

        if (tp + fp + fn + tn) > 0:
            random_accuracy = ((tn + fp) * (tn + fn) + (fn + tp) * (fp + tp)) / (tp + fp + fn + tn) ** 2
            kappa = (accuracy - random_accuracy) / (1 - random_accuracy)
        else:
            kappa = random_accuracy = None

        classes.append({"accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "random_acc": random_accuracy,
                        "fscore": fscore,
                        "specificity": specificity,
                        "kappa": kappa,
                        "total": total})

    return overall_accuracy, classes

def create_roc_table(conn, drop=False):
    cur = conn.cursor()

    if drop:
        cur.execute("DROP TABLE IF EXISTS roc CASCADE")

    # create Table SQL
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS roc (
            log_id serial PRIMARY KEY,
            run text,
            descr text,
            step int,
            epoch int,
            class int,
            scores text,
            targets text,
            obs text
        );"""

    cur.execute(create_table_sql)
    conn.commit()
    cur.close()

def log_roc_table(conn, scores,targets,obs,runname, descr="", step=None, epoch=None):

    cur = conn.cursor()

    _, n_classes = scores.shape

    obs_pkl = pickle.dumps(obs)
    for c in range(n_classes):
        sc = scores[:, c]
        ta = targets[:, c]


        scores_pkl = pickle.dumps(sc)
        targets_pkl = pickle.dumps(ta)


        query = """
            INSERT INTO roc (
                run,
                descr,
                step,
                epoch,
                class,
                scores,
                targets,
                obs)
             VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s);"""

        cur.execute(query,
                    [runname,
                     descr,
                     step,
                     epoch,
                     c,
                     scores_pkl,
                     targets_pkl,
                     obs_pkl])

    conn.commit()
    cur.close()

def create_cm_table(conn, drop=False):
    cur = conn.cursor()

    if drop:
        cur.execute("DROP TABLE IF EXISTS cm CASCADE")

    # create Table SQL
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS cm (
            log_id serial PRIMARY KEY,
            run text,
            descr text,
            step int,
            epoch int,
            cm text
        );"""

    cur.execute(create_table_sql)
    conn.commit()
    cur.close()

def log_cm_table(cm, conn, run, step, epoch, descr=""):

    cur = conn.cursor()

    cm_pkl = pickle.dumps(cm)

    query = """
    INSERT INTO cm (
        run,
        descr,
        step,
        epoch,
        class,
        cm)
     VALUES
        (%s, %s, %s, %s, %s, %s);"""

    cur.execute(query,
                [run,
                 descr,
                 step,
                 epoch,
                 cm_pkl])

    conn.commit()
    cur.close()

def create_eval_table(conn, drop=False):
    cur = conn.cursor()

    if drop:
        cur.execute("DROP TABLE IF EXISTS eval CASCADE")

    # create Table SQL
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS eval (
            log_id serial PRIMARY KEY,
            run text,
            descr text,
            step int,
            epoch int,
            class int,
            accuracy double precision,
            prec double precision,
            recall double precision,
            fscore double precision,
            kappa double precision,
            total int
        );"""

    cur.execute(create_table_sql)
    conn.commit()
    cur.close()


def log_in_db(data, conn, run, step, epoch, descr=""):
    d = pd.DataFrame(data)
    cur = conn.cursor()

    for cl in range(len(data)):
        d = data[cl]

        query = """
        INSERT INTO eval (
            run,
            descr,
            step,
            epoch,
            class,
            accuracy,
            prec,
            recall,
            fscore,
            kappa,
            total)
         VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""

        cur.execute(query,
                    [run,
                     descr,
                     step,
                     epoch,
                     cl,
                     d["accuracy"],
                     d["precision"],
                     d["recall"],
                     d["fscore"],
                     d["kappa"],
                     d["total"]])

    conn.commit()
    cur.close()


""" Tests """


def test1():
    # http: // www.dataschool.io / simple - guide - to - confusion - matrix - terminology /
    cnf = [[50., 10.],
           [5., 100.]]

    overall_accuracy, (c1, c2) = class_evaluation(cnf)

    print("Test 1")
    print()
    print("overall accurcacy correct:", np.round(overall_accuracy, 2) == 0.91)
    print("c2 accuracy", np.round(c2["accuracy"], 2) == 0.91)
    print("c2 recall", np.round(c2["recall"], 2) == 0.95)
    print("c2 precision", np.round(c2["precision"], 2) == 0.91)


def test2():
    # https: // www.researchgate.net / post / Can_someone_help_me_to_calculate_accuracy_sensitivity_of_a_66_confusion_matrix
    cnf = np.array([[1971, 19, 1, 8, 0, 1],
                    [16, 1940, 2, 23, 9, 10],
                    [8, 3, 1891, 87, 0, 11],
                    [2, 25, 159, 1786, 16, 12],
                    [0, 24, 4, 8, 1958, 6],
                    [11, 12, 29, 11, 11, 1926]])

    overall_accuracy, (c1, c2, c3, c4, c5, c6) = class_evaluation(cnf)

    print()
    print("Test 2")
    print("overall accurcacy correct:", np.round(overall_accuracy, 4) == 0.956)
    print()
    print("c1 accuracy", np.round(c1["accuracy"], 4) == 0.9943)
    print("c1 precision", np.round(c1["precision"], 4) == 0.9816)
    print("c1 recall", np.round(c1["recall"], 4) == 0.9855)
    print("c1 fscore", np.round(c1["fscore"], 4) == 0.9835)
    print()
    print("c2 accuracy", np.round(c2["accuracy"], 4) == 0.9877)
    print("c2 precision", np.round(c2["precision"], 4) == 0.9590)
    print("c2 recall", np.round(c2["recall"], 4) == 0.9700)
    print("c2 fscore", np.round(c2["fscore"], 4) == 0.9645)
    print()
    print("c3 accuracy", np.round(c3["accuracy"], 4) == 0.9742)
    print("c3 precision", np.round(c3["precision"], 4) == 0.9065)
    print("c3 recall", np.round(c3["recall"], 4) == 0.9455)
    print("c3 fscore", np.round(c3["fscore"], 4) == 0.9256)


def test3():
    # http://text-analytics101.rxnlp.com/2014/10/computing-precision-and-recall-for.html

    cnf = np.array([[30, 20, 10],
                    [50, 60, 10],
                    [20, 20, 80]]).T

    overall_accuracy, (c1, c2, c3) = class_evaluation(cnf)

    print()
    print("Test 3")
    print("c1 precision", np.round(c1["precision"], 1) == 0.5)
    print("c1 recall", np.round(c1["recall"], 1) == 0.3)
    print()
    print("c2 precision", np.round(c2["precision"], 1) == 0.5)
    print("c2 recall", np.round(c2["recall"], 1) == 0.6)


def test4():
    # http://www.springer.com/cda/content/document/cda_downloaddocument/Cohens+Kappa.pdf?SGWID=0-0-45-1426183-p175274210
    cnf = np.array([[31, 1],
                    [3, 122]])

    overall_accuracy, (c1, c2) = class_evaluation(cnf)

    print()
    print("Test 4")
    print("c1 kappa", np.round(c1["kappa"], 2) == 0.92)


def main():
    test1()
    test2()
    test3()
    test4()


if __name__ == '__main__':
    main()
