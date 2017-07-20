#!/home/russwurm/anaconda2/envs/field2/bin/python

#import psycopg2
import cPickle as pickle
import os
import numpy as np

class Dataloader:

    def __init__(self, datafolder, nbatches=None, batchsize=None):

        self.datafolder=datafolder

        # take less bat
        if nbatches is None:
            self.nbatches=len(os.listdir(datafolder))
        else:
            self.nbatches = nbatches

        self.epoch=0
        self.batch=0

        self.databatchsize, self.maxobs, self.nfeatures, self.nclasses = self.query_datashape()


        if batchsize is None: # no other specified batchsize
            self.batchsize = self.databatchsize #  take batch size from the data
        else:
            if batchsize % self.databatchsize == 0: # specified batchsize must be a multiple of databatchsize
                self.batchsize = batchsize
            else:
                raise ValueError("specified batchsize ({}) must be a multiple of batchsize of data ({})".format(batchsize, self.databatchsize))


        # compatibility...
        self.batch_size = self.batchsize
        self.num_feat = self.nfeatures
        self.num_batches = self.nbatches

        return None

    def query_datashape(self):
        x, y, obs = self.get_databatch(advance=False)
        batchsize, maxobs, nfeatures = x.shape
        batchsize, _,  nclasses = y.shape
        return batchsize, maxobs, nfeatures, nclasses

    def new_epoch(self):
        self.epoch += 1
        self.batch = 0

    def get_databatch(self, advance=True):
        if self.batch >= self.nbatches:
            self.new_epoch()
        with open(os.path.join(self.datafolder, "b{}.pkl".format(self.batch)), "rb") as f:
            data = pickle.load(f)
        if advance: self.batch += 1
        return data

    def next_batch(self):
        return self.get_databatch()


#if __name__ == '__main__':