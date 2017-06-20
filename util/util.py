import pandas
import re
import cPickle as pickle
import os

def write_status_file(savedir, step, epoch, eta=None, train_xentropy=None, test_xentropy=None, train_oa=None, test_oa=None):
    with open(savedir + "/steps.txt", "w") as f:
        f.write("{} {} {} {} {} {} {}".format(step,epoch,eta,train_xentropy,test_xentropy,train_oa,test_oa))

def read_status_file(init_from):
    with open(init_from + "/steps.txt", "r") as f:
        line = f.read()
        step_, epoch_ = line.split(" ")[0:2]
    return int(step_), int(epoch_)

def params2name(l, r, d, f):
    return "{}l{}r{}d{}f".format(l, r, int(d * 100), f)

def name2params(name):
    regex = "[0-9]+[a-z]"
    exp = re.findall(regex, name)
    ls = []
    for e in exp:
        ls.append(int(re.sub("\D", "", e)))
    return ls

def getargs(runname):
    return pickle.load(open(os.path.join(runname, "args.pkl"), "r"))

def query_column_names(conn, table, schema='public'):
    sql="""
    SELECT column_name
    FROM
    information_schema.columns
    WHERE
    table_schema=\'{0}\'
    AND
    table_name=\'{1}\'
    """.format(schema, table)

    return pd.read_sql(sql,conn)["column_name"].values
