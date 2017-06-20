#!/home/russwurm/anaconda2/envs/field2/bin/python

import psycopg2
import pandas as pd
import datetime
import numpy as np
import cPickle as pickle
import os
from util.multiprocessing_tools import parmap
import StringIO


def pad(arr, n_max_obs):
    """
        Helper function padds array up zo n_max_obs in direction axis=1

    """

    i, obs = arr.shape
    pa = np.zeros((i, n_max_obs - obs))
    return np.concatenate((arr, pa), axis=1).T

class DataLoader:

    def __init__(self,
                 conn = None,
                 batch_size=50,
                 debug=False,
                 sql_where="",
                 random_seed=1,
                 do_shuffle=True,
                 do_init_shuffle=True,
                 n_max_obs=26,
                 fn_new_epoch_callback=None,
                 pack_size=None,
                 tablename="grid_batches",
                 packed_table = None,
                 localdir=None):

        ''' constructor '''
        if debug: print("initializing DataLoader")
        # selector for database
        # e.g. 'is_train=True' or '' for no selection
        self.sql_where = sql_where

        self.n_max_obs = n_max_obs

        self.tablename = tablename

        self.packed_table = packed_table

        if debug: print("\testablishing database connection...")
        self.conn = conn

        self.debug = debug
        self.do_shuffle = do_shuffle

        self.batch_size = batch_size

        # callback function for new epoch
        self.fn_new_epoch_callback = fn_new_epoch_callback

        if debug: print("querying ids from db")
        np.random.seed(random_seed)
        self.ids = pd.read_sql_query("select field_id as id from {} {}".format(tablename,sql_where), self.conn)["id"].values

        self.pack_size = pack_size
        if pack_size is not None and (batch_size % pack_size != 0):
            print "Warning: batchsize is not a multiple of packsize ignoring packing"
            pack_size = None

        if pack_size is not None:
            print "Using packing indices..."
            packs = pd.read_sql_query("select id from {} {}".format(packed_table,sql_where), self.conn)["id"].values.astype(int)

            self.packs = packs[~np.isnan(packs)]

        if do_init_shuffle:
            np.random.shuffle(self.ids)

            if pack_size is not None:
                np.random.shuffle(self.packs)

        if debug: print("\tquery total number of fields...")
        self.num_feat = len(self.ids)

        if debug: print("\t\tnum_feat = " + str(self.num_feat))

        self.classes = pd.read_sql_query("select name from label_lookup", self.conn)["name"].values
        self.n_classes = len(self.classes)

        self.generate_batches(self.ids, self.batch_size)

        self.batch = 0
        self.epoch = 0

        self.localdir = localdir
        # if local dir is specified
        if localdir is not None:
            # if no folder already exists, make folder and download
            if not os.path.exists(localdir):
                os.makedirs(localdir)
                self.download(localdir)



        return None

    def generate_batches(self, ids, batch_size):
        # drop last fields to fit data to even batches
        n_too_many = len(ids) % batch_size
        ids_ = ids[0:ids.size - n_too_many]

        # batches defined as list of ids
        self.batches = ids_.reshape(-1, batch_size)

        # batches as list of packs
        if self.pack_size is not None:

            n_too_many = len(self.packs) % (batch_size/self.pack_size)
            packs = self.packs[0:self.packs.size - n_too_many]

            self.batches_pack = packs.reshape(-1, batch_size/self.pack_size)

        self.num_batches,_ = self.batches.shape


    def reset(self):
        self.epoch=0
        self.batch=0

    def new_epoch(self):
        self.epoch += 1
        self.batch = 0

        if self.do_shuffle:
            np.random.shuffle(self.ids)

            self.generate_batches(self.ids, self.batch_size)

        # callback function
        if self.fn_new_epoch_callback is not None:
            self.fn_new_epoch_callback()

    def next_batch_packed(self):
        """

        get batches from <packed_table> containing batches arranged in packs of pack_size features
        this accelarated query and processing massively

        :return:
        """
        if self.batch >= self.num_batches:
            self.new_epoch()
        
        pack_ids = self.batches_pack[self.batch]

        if len(pack_ids) > 1:
            where = "id in {}".format(tuple(pack_ids))
        else:
            where = "id={}".format(pack_ids[0])

        sql = """
                    select data
                    from {}
                    where
                    {}

                """.format(self.packed_table, where)

        s_sql = datetime.datetime.now()
        dat = pd.read_sql_query(sql, self.conn)


        e_sql = datetime.datetime.now()

        buff_list = dat.data.values

        x, y, n_obs = pickle.load(StringIO.StringIO(buff_list[0]))
        for i in range(1,len(buff_list)):
            x_,y_,n_obs_ = pickle.load(StringIO.StringIO(buff_list[i]))
            x = np.append(x,x_, axis=0)
            y = np.append(y, y_, axis=0)
            n_obs = np.append(n_obs, n_obs_, axis=0)

        e_pkl = datetime.datetime.now()
        if self.debug:
            dt_sql = e_sql - s_sql
            dt_pkl = e_pkl - e_sql
            dt_total = e_pkl - s_sql
            print("next_batch time summary:")
            msg = "total time elapsed: %d ms (sql: %d ms, unpickle %d ms)" % (
                dt_total.total_seconds() * 1000, dt_sql.total_seconds() * 1000,
                dt_pkl.total_seconds() * 1000)
            print(msg)

        self.batch += 1
        return x, y, n_obs

    def next_batch(self):
        if (self.localdir is not None):
            return self.next_batch_local()
        if self.pack_size is None:
            return self.next_batch_unpacked()
        else:
            return self.next_batch_packed()

    def download(self, downloaddir):
        self.reset()

        print "downloading {} batches to {}".format(self.num_batches, downloaddir)

        if not os.path.exists(downloaddir):
            os.makedirs(downloaddir)

        #h5file = tables.open_file(downloaddir+"/data.h5", "w", driver="H5FD_CORE")

        for i in range(self.num_batches):
            data = self.next_batch_unpacked()
            #a = h5file.create_array(h5file.root, "b{}".format(i),(data))
            with open(os.path.join(downloaddir,"b{}.pkl".format(i)), "wb") as f:
                pickle.dump(data, f, protocol=2)

        self.reset()

    def next_batch_local(self):
        if self.batch >= self.num_batches:
            self.new_epoch()
        with open(os.path.join(self.localdir,"b{}.pkl".format(self.batch)), "rb") as f:
            data = pickle.load(f)
        self.batch += 1
        return data

    def next_batch_unpacked(self):
        ''' returns next batch from an intermediate database table containing pickled X and y.

            1. query data from database

        '''


        if self.batch >= self.num_batches:
            self.new_epoch()

        # split batch in subgroups of 10 to fit sql in clause
        if self.batch_size > 50:
            batch_groups = np.array_split(self.batches[self.batch], self.batch_size/50)
        else:
            batch_groups = self.batches[self.batch].reshape((1,-1))

        where = "field_id IN "
        for g in batch_groups:
            where += str(tuple(g))

        # add OR between tuples
        where = where.replace(")(", ")\n OR field_id IN (")


        sql = """
            select field_id, x_data, y_data, n_obs
            from {}
            where
            {}

        """.format(self.tablename, where)


        s_sql = datetime.datetime.now()
        dat = pd.read_sql_query(sql, self.conn)

        e_sql = datetime.datetime.now()
        x_data = dat["x_data"]
        y_data = dat["y_data"]
        seq_lengths = dat["n_obs"].values

        # apply this function on each element of queried Dataframe
        # pads data to n_max_obs with zeros
        def unpickle(str):

            if isinstance(str, basestring):
                d = pickle.loads(str)
            else:
                d = pickle.load(StringIO.StringIO(str))

            i, obs = d.shape
            pad = np.zeros((i, self.n_max_obs - obs))

            # Transpose to match [batchsize x obs x n_input] format of tf.dynamic_rnn
            return np.concatenate((d, pad), axis=1).T


        x = np.array(x_data.apply(unpickle).tolist())
        y = np.array(y_data.apply(unpickle).tolist())


        #actually slower...
        #x = parmap(unpickle, x_data.tolist())
        #y = parmap(unpickle, y_data.tolist())
        #e = datetime.datetime.now()


        #dt_apply = (m-s).total_seconds()
        #dt_parmap = (e - m).total_seconds()

        e_pkl = datetime.datetime.now()
        if self.debug:
            dt_sql = e_sql - s_sql
            dt_pkl = e_pkl - e_sql
            dt_total = e_pkl - s_sql
            print("next_batch time summary:")
            msg = "total time elapsed: %d ms (sql: %d ms, unpickle %d ms)" % (
                dt_total.total_seconds() * 1000, dt_sql.total_seconds() * 1000,
                dt_pkl.total_seconds() * 1000)
            print(msg)

        self.batch += 1
        return x, y, seq_lengths

    """ pack functionality """

    def create_new_packed_table(self, tablename, train_col="is_train"):
        """ drop old table and create new one """
        cur = self.conn.cursor()

        cur.execute("DROP TABLE IF EXISTS %s CASCADE" % (tablename))

        # create Table SQL
        create_table_sql = """
            CREATE TABLE {0} (
                id SERIAL PRIMARY KEY,
                {1} BOOLEAN,
                data bytea
            );
        """.format(tablename, train_col)

        print "create new table {}".format(tablename)
        cur.execute(create_table_sql)
        conn.commit()
        cur.close()

    def pack(self, is_train, commit_every=100, tablename="tmp", train_col="is_train", drop=False):
        "is train -> boolean written in istrain column"

        if drop:
            self.create_new_packed_table(tablename, train_col=train_col)

        cur = self.conn.cursor()

        self.batch = 0
        while self.epoch < 1:

            X, y, n_obs = self.next_batch()

            buff = pickle.dumps([X, y, n_obs], protocol=2)

            query = """
                                    INSERT INTO {0} (
                                        {1},
                                        data)
                                        VALUES
                                        (%s, %s)""".format(tablename,train_col)

            # insert to_db dict to database
            cur.execute(query, [is_train, psycopg2.Binary(buff)])

            if self.batch % commit_every == 0:
                conn.commit()

            print "packing {}/{}".format(self.batch, self.num_batches)

        conn.commit()
        cur.close()

    def query_column_names(conn, table, schema='public'):
        sql = """
        SELECT column_name
        FROM
        information_schema.columns
        WHERE
        table_schema=\'{0}\'
        AND
        table_name=\'{1}\'
        """.format(schema, table)

        return pd.read_sql(sql, conn)["column_name"].values

    def query_tables(self):
        sql = """
        SELECT
        table_name
        FROM
        INFORMATION_SCHEMA.TABLES
        WHERE
        TABLE_TYPE = 'BASE TABLE'
        AND
        table_schema = 'public'
        AND
        table_catalog = 'dbBayField'
        """
        return pd.read_sql(sql,self.conn)["table_name"].values

if __name__ == '__main__':
    conn = psycopg2.connect('postgres://russwurm:dbfieldpassword@localhost/dbBayField')
    import matplotlib.pyplot as plt

    batch_size = 2000

    train_dataloader = DataLoader(conn=conn, batch_size=500, debug=True, do_shuffle=True,
                                  sql_where="where is_train0=True", tablename="raster_label_fields", localdir="tmp/data")

    train_dataloader.num_feat

    if False:
        train_dataloader = DataLoader(conn=conn, batch_size=batch_size, sql_where="where is_train{}=True".format(0),
                                      debug=False,
                                      do_shuffle=False, do_init_shuffle=True, tablename="raster_label_fields")

    if False:
        print "download"
        #train_dataloader.download("tmp/data/")
    if True:
        for i in range(2*train_dataloader.num_batches):
            x,y,seq = train_dataloader.next_batch_local()
            print "batch {}, epoch {}".format(train_dataloader.batch, train_dataloader.epoch)

    if False:
        train_dataloader.pack(drop=False, tablename=tablename, is_train=False, train_col=train_col)

    if False:
        # batchsize -> packsize

        train_col = "is_train_random"
        tablename = "packed_batches_random"

        train_dataloader = DataLoader(conn=conn, batch_size=500, debug=True, do_shuffle=True, sql_where="where {}=True".format(train_col), tablename="raster_label_fields")
        train_dataloader.pack(drop=True, tablename=tablename, is_train=True, train_col=train_col)

        test_dataloader = DataLoader(conn=conn, batch_size=500, debug=True, do_shuffle=True, sql_where="where {}=False".format(train_col), tablename="raster_label_fields")
        test_dataloader.pack(drop=False, tablename=tablename, is_train=False, train_col=train_col)

        train_col = "is_train_eastwest"
        tablename = "packed_batches_eastwest"

        train_dataloader = DataLoader(conn=conn, batch_size=500, debug=True, do_shuffle=True, sql_where="where {}=True".format(train_col), tablename="raster_label_fields")
        train_dataloader.pack(drop=True, tablename=tablename, is_train=True, train_col=train_col)

        test_dataloader = DataLoader(conn=conn, batch_size=500, debug=True, do_shuffle=True, sql_where="where {}=False".format(train_col), tablename="raster_label_fields")
        test_dataloader.pack(drop=False, tablename=tablename, is_train=False, train_col=train_col)


    if False:
        print "with pack"

        dataloader = DataLoader(conn=conn, batch_size=batch_size, debug=True, do_shuffle=True, pack_size=500, packed_table="packed_batches", tablename="raster_label_fields")

        X, y, n_obs = dataloader.next_batch_packed()
        X, y, n_obs = dataloader.next_batch_packed()
        X, y, n_obs = dataloader.next_batch_packed()
        X, y, n_obs = dataloader.next_batch_packed()
        X, y, n_obs = dataloader.next_batch_packed()

    if False:
        print "grid_batches without pack"

        dataloader = DataLoader(conn=conn, batch_size=batch_size, debug=True, do_shuffle=True)

        X, y, n_obs = dataloader.next_batch()
        X, y, n_obs = dataloader.next_batch()
        X, y, n_obs = dataloader.next_batch()
        X, y, n_obs = dataloader.next_batch()
        X, y, n_obs = dataloader.next_batch()

    if False:
        print "grid_batches2 without pack"

        dataloader = DataLoader(conn=conn, batch_size=batch_size, debug=True, do_shuffle=True, tablename="grid_batches2")

        X, y, n_obs = dataloader.next_batch()
        X, y, n_obs = dataloader.next_batch()
        X, y, n_obs = dataloader.next_batch()
        X, y, n_obs = dataloader.next_batch()
        X, y, n_obs = dataloader.next_batch()

    if False:
        dataloader = DataLoader(conn=conn, batch_size=2000, debug=True, do_shuffle=True)

        dataloader.save_local("tmp")
