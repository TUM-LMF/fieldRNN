import tensorflow as tf
import cPickle as pickle
import rnn_model
import cnn_model
from dataloader import Dataloader
#import psycopg2
import os
import datetime
import numpy as np
import argparse
from util.util import write_status_file, read_status_file, params2name
import sys

"""
This file contains three functions.

    main() provides a shell interface for training from CLI

    train_{rnn|cnn} are called by main to perform the training on {rnn|cnn}_model Tensorflow Graphs


Dependencies:
    Dataloader.py
    rnn_model.py
    cnn_model.py
"""

def train_rnn(model,
          train_dataloader,
          test_dataloader,
          savedir="save/tmp",
          max_epoch=None,
          log_every=20,
          save_every=100,
          print_every=5,
          init_from=None,
          max_ckpts_to_keep=5,
          ckpt_every_n_hours=10000,
          allow_gpu_mem_growth=True,
          gpu_memory_fraction=0.3,
          **kwargs):
    """
    This function performs the training operation on a tensorflow rnn_model.py model

    :param model:               rnn_model object containing tensorflow graph
    :param train_dataloader:    DataLoader object for Training data
    :param test_dataloader:     DataLoader object for Testing data
    :param savedir:             directory to store event and save files
    :param max_epoch:           hard maximum for number of epochs
    :param log_every:           Frequency of TensorFlow summary recordings
    :param save_every:          checkpoint save frequency
    :param print_every:         console log frequency
    :param init_from:           initialize weights from checkpoint files
    :param max_ckpts_to_keep:   tf.train.Saver: maximum number of checkpoint files
    :param ckpt_every_n_hours:  save every n hours
    :param allow_gpu_mem_growth:dynamic growth of gpu vram
    :param gpu_memory_fraction: hard upper limit for gpu vram
    :return: True if success
    """

    terminate = False

    if not os.path.exists(savedir + "/train"):
        os.makedirs(savedir + "/train")

    if not os.path.exists(savedir + "/test"):
        os.makedirs(savedir + "/test")

    # save list of classes
    #np.save(os.path.join(savedir, "classes.npy"), train_dataloader.classes)

    # dump pickle args for loading
    with open(os.path.join(savedir, "args.pkl"), "wb") as f:
        pickle.dump(model.args, f)
    # dump human readable args
    open(os.path.join(savedir, "args.txt"), "w").write(str(model.args))

    train_summary_writer = tf.summary.FileWriter(savedir + "/train", graph=tf.get_default_graph())
    test_summary_writer = tf.summary.FileWriter(savedir + "/test", graph=tf.get_default_graph())

    saver = tf.train.Saver(max_to_keep=max_ckpts_to_keep, keep_checkpoint_every_n_hours=ckpt_every_n_hours)

    step = 0
    t_last = datetime.datetime.now()

    total_cm_train = total_cm_test = np.zeros((model.n_classes, model.n_classes))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = allow_gpu_mem_growth
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    config.allow_soft_placement = True
    config.log_device_placement = False

    train_cross_entropy = None
    test_cross_entropy = None
    eta = None

    print("start")
    with tf.Session(config=config) as sess:

        sess.run([model.init_op])

        if init_from is not None:
            if os.path.exists(init_from):

                try:

                    ckpt = tf.train.get_checkpoint_state(init_from)
                    print("restoring model from %s" % ckpt.model_checkpoint_path)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    step, epoch = read_status_file(init_from)
                    train_dataloader.epoch = epoch

                except:
                    print "error at {} ignoring".format(init_from)
                    init_from = None
                    pass

        i = 0
        while (train_dataloader.epoch < max_epoch) or terminate:
            i += 1

            # step as number of features -> invariant to changes in batch size
            step += train_dataloader.batch_size

            s_db = datetime.datetime.now()
            X, y, seq_lengths = train_dataloader.next_batch()
            e_db = datetime.datetime.now()

            feed = {model.X: X, model.y_: y, model.seq_lengths: seq_lengths}

            # training step
            _, cm = sess.run([model.train_op, model.confusion_matrix], feed_dict=feed)
            #total_cm_train += cm

            e_tr = datetime.datetime.now()

            dt_db = e_db - s_db
            dt_tr = e_tr - e_db

            field_per_s = train_dataloader.batch_size / (datetime.datetime.now() - t_last).total_seconds()
            # approximate calculation time
            approx_calc_time = (((max_epoch * train_dataloader.num_feat) - step) / field_per_s)
            eta = datetime.datetime.now() + datetime.timedelta(seconds=approx_calc_time)

            t_last = datetime.datetime.now()

            if i % print_every == 0:
                train_cross_entropy = sess.run(model.cross_entropy, feed_dict=feed)
                msg = "Training: Iteration {}, feature {}, epoch {}, batch {}/{}: xentr {:.2f} " \
                      "(time: db {}ms; train {}ms, {} feat/s, eta: {})".format(
                    i,
                    step,
                    train_dataloader.epoch,
                    train_dataloader.batch,
                    train_dataloader.num_batches,
                    train_cross_entropy,
                    int(dt_db.total_seconds() * 1000),
                    int(dt_tr.total_seconds() * 1000),
                    int(field_per_s),
                    eta.strftime("%d.%b %H:%M")
                )
                print(msg)

            if i % log_every == 0:  # Record summaries and test-set accuracy
                # record with train data
                summary = sess.run(model.merge_summary_op, feed_dict=feed)
                train_summary_writer.add_summary(summary, step)

                # record with test data
                X, y, seq_lengths = test_dataloader.next_batch()
                feed = {model.X: X, model.y_: y, model.seq_lengths: seq_lengths}
                test_cross_entropy, summary = sess.run([model.cross_entropy, model.merge_summary_op], feed_dict=feed)
                #total_cm_test += cm
                test_summary_writer.add_summary(summary, step)

                with tf.name_scope('performance'):
                    # custom summaries
                    summary = tf.Summary(value=[
                        tf.Summary.Value(tag="fields_per_sec", simple_value=field_per_s),
                        tf.Summary.Value(tag="query_time_sec", simple_value=dt_db.total_seconds()),
                        tf.Summary.Value(tag="train_time_sec", simple_value=dt_tr.total_seconds())
                    ])
                train_summary_writer.add_summary(summary, step)

                print("writing summary")

            if i % save_every == 0:
                if not os.path.exists(savedir):
                    os.makedirs(savedir)

                last_checkpoint = os.path.join(savedir, 'model.ckpt')
                saver.save(sess, last_checkpoint, global_step=step)

                write_status_file(savedir, step, train_dataloader.epoch)

                # update task table
                if "update_callback" in kwargs.keys() and (train_cross_entropy is not None) and (test_cross_entropy is not None) and (eta is not None):
                    kwargs["update_callback"](step, train_dataloader.epoch, train_cross_entropy, test_cross_entropy, eta.strftime("%d.%b %H:%M"))

        # save very last state
        last_checkpoint = os.path.join(savedir, 'model.ckpt')
        saver.save(sess, last_checkpoint, global_step=step)
        write_status_file(savedir, step, train_dataloader.epoch)

        # update task table
        if "update_callback" in kwargs.keys() and (train_cross_entropy is not None) and (
            test_cross_entropy is not None) and (eta is not None):
            kwargs["update_callback"](step, train_dataloader.epoch, train_cross_entropy, test_cross_entropy,
                                      eta.strftime("%d.%b %H:%M"))

    return True

def train_cnn(model,
          train_dataloader,
          test_dataloader,
          savedir="save/tmp",
          max_epoch=None,
          log_every=20,
          save_every=100,
          print_every = 5,
          init_from=None,
          max_ckpts_to_keep = 5,
          ckpt_every_n_hours=10000,
          allow_gpu_mem_growth=True,
          gpu_memory_fraction=0.3,
          **kwargs):
    """
    This function performs the training operation on a tensorflow rnn_model.py model

    :param model:               rnn_model object containing tensorflow graph
    :param train_dataloader:    DataLoader object for Training data
    :param test_dataloader:     DataLoader object for Testing data
    :param savedir:             directory to store event and save files
    :param max_epoch:           hard maximum for number of epochs
    :param log_every:           Frequency of TensorFlow summary recordings
    :param save_every:          checkpoint save frequency
    :param print_every:         console log frequency
    :param init_from:           initialize weights from checkpoint files
    :param max_ckpts_to_keep:   tf.train.Saver: maximum number of checkpoint files
    :param ckpt_every_n_hours:  save every n hours
    :param allow_gpu_mem_growth:dynamic growth of gpu vram
    :param gpu_memory_fraction: hard upper limit for gpu vram
    :return: True if success
    """

    terminate = False

    if not os.path.exists(savedir + "/train"):
        os.makedirs(savedir + "/train")

    if not os.path.exists(savedir + "/test"):
        os.makedirs(savedir + "/test")

    # save list of classes
    #np.save(os.path.join(savedir, "classes.npy"), train_dataloader.classes)

    # dump pickle args for loading
    with open(os.path.join(savedir, "args.pkl"), "wb") as f:
        pickle.dump(model.args, f)
    # dump human readable args
    open(os.path.join(savedir, "args.txt"), "w").write(str(model.args))

    train_summary_writer = tf.summary.FileWriter(savedir + "/train", graph=tf.get_default_graph())
    test_summary_writer = tf.summary.FileWriter(savedir + "/test", graph=tf.get_default_graph())

    saver = tf.train.Saver(max_to_keep=max_ckpts_to_keep, keep_checkpoint_every_n_hours=ckpt_every_n_hours)

    step = 0
    t_last = datetime.datetime.now()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = allow_gpu_mem_growth
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    config.allow_soft_placement = True
    config.log_device_placement = False

    train_cross_entropy = None
    test_cross_entropy = None
    eta = None

    print("start")
    with tf.Session(config=config) as sess:

        sess.run([model.init_op])

        if init_from is not None:
            if os.path.exists(init_from):

                try:

                    ckpt = tf.train.get_checkpoint_state(init_from)
                    print("restoring model from %s" % ckpt.model_checkpoint_path)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    step, epoch = read_status_file(init_from)
                    train_dataloader.epoch = epoch

                except:
                    print "error at {} ignoring".format(init_from)
                    init_from = None
                    pass

        i = 0
        while (train_dataloader.epoch < max_epoch) or terminate:
            i += 1

            # step as number of features -> invariant to changes in batch size
            step += train_dataloader.batch_size

            s_db = datetime.datetime.now()
            X, y, seq_lengths = train_dataloader.next_batch()
            e_db = datetime.datetime.now()

            """ unroll data """
            X, y = cnn_model.unroll(X, y, seq_lengths)

            feed = {model.X: X, model.y: y, model.batch_size:X.shape[0]}

            # training step
            _= sess.run(model.train_op, feed_dict=feed)

            e_tr = datetime.datetime.now()

            dt_db = e_db - s_db
            dt_tr = e_tr - e_db

            field_per_s = train_dataloader.batch_size / (datetime.datetime.now() - t_last).total_seconds()
            # approximate calculation time
            approx_calc_time = (((max_epoch*train_dataloader.num_feat)-step) / field_per_s)
            eta = datetime.datetime.now() + datetime.timedelta(seconds=approx_calc_time)

            t_last = datetime.datetime.now()

            if i % print_every == 0:
                train_cross_entropy = sess.run(model.cross_entropy, feed_dict=feed)
                msg =  "Training: Iteration {}, feature {}, epoch {}, batch {}/{}: xentr {:.2f} " \
                      "(time: db {}ms; train {}ms, {} feat/s, eta: {})".format(
                    i,
                    step,
                    train_dataloader.epoch,
                    train_dataloader.batch,
                    train_dataloader.num_batches,
                    train_cross_entropy,
                    int(dt_db.total_seconds() * 1000),
                    int(dt_tr.total_seconds() * 1000),
                    int(field_per_s),
                    eta.strftime("%d.%b %H:%M")
                )
                print(msg)

            if i % log_every == 0:  # Record summaries and test-set accuracy
                # record with train data
                summary, test_cross_entropy = sess.run([model.merge_summary_op, model.cross_entropy], feed_dict=feed)
                train_summary_writer.add_summary(summary, step)

                # record with test data
                X, y, seq_lengths = test_dataloader.next_batch()

                X, y = cnn_model.unroll(X, y, seq_lengths)

                feed = {model.X: X, model.y: y, model.batch_size: X.shape[0]}
                summary = sess.run(model.merge_summary_op, feed_dict=feed)
                test_summary_writer.add_summary(summary, step)

                with tf.name_scope('performance'):
                # custom summaries
                    summary = tf.Summary(value=[
                        tf.Summary.Value(tag="fields_per_sec", simple_value=field_per_s),
                        tf.Summary.Value(tag="query_time_sec", simple_value=dt_db.total_seconds()),
                        tf.Summary.Value(tag="train_time_sec", simple_value=dt_tr.total_seconds())
                    ])
                train_summary_writer.add_summary(summary, step)

                print("writing summary")

            if i % save_every == 0:
                if not os.path.exists(savedir):
                    os.makedirs(savedir)

                last_checkpoint = os.path.join(savedir, 'model.ckpt')
                saver.save(sess, last_checkpoint, global_step=step)

                with open(savedir+"/steps.txt","w") as f:
                    f.write("%s %s" % (step, train_dataloader.epoch))

                # update task table
                if "update_callback" in kwargs.keys() and (train_cross_entropy is not None) and (test_cross_entropy is not None) and (eta is not None):
                    kwargs["update_callback"](step, train_dataloader.epoch, train_cross_entropy, test_cross_entropy, eta.strftime("%d.%b %H:%M"))

        # save very last state
        last_checkpoint=os.path.join(savedir,'model.ckpt')
        saver.save(sess,last_checkpoint, global_step=step)
        with open(savedir+"/steps.txt","w") as f:
            f.write("{} {}".format(step,train_dataloader.epoch))

        if "update_callback" in kwargs.keys() and (train_cross_entropy is not None) and (
            test_cross_entropy is not None) and (eta is not None):

            kwargs["update_callback"](step, train_dataloader.epoch, train_cross_entropy, test_cross_entropy,
                                      eta.strftime("%d.%b %H:%M"))

    return True

def main():
    parser = argparse.ArgumentParser(description='Train neural network.')

    parser.add_argument('layers', type=int, help='number of layers')
    parser.add_argument('cells', type=int, help='number of rnn cells, as multiple of 55')
    parser.add_argument('dropout', type=float, help='dropout keep probability')
    parser.add_argument('fold', type=int, help='select training/evaluation fold to use')

    parser.add_argument('maxepoch', type=int, help="maximum epochs")
    parser.add_argument('batchsize', type=int, help="batchsize")

    parser.add_argument('--savedir', type=str, default="save/tmp", help='directory to save the run')
    #parser.add_argument('--batchsize', type=int, default=500, help="batchsize")
    parser.add_argument('--gpu', '-g', type=str, default=None, help='visible gpu')

    parser.add_argument('--model', type=str, help="Neural network architecture. 'lstm', 'rnn' or 'cnn'", default='lstm')
    parser.add_argument('--max_ckpts_to_keep', '-c', type=int ,default=10, help='number of checkpoints to keep')
    parser.add_argument('--ckpt_every_n_hours', '-t', type=float ,default=0.5, help='save checkpoint every n hours')
    parser.add_argument('--save_every', '-S', type=int, default=100, help='iteration to save a checkpoint')

    parser.add_argument('--summary_every', '-s', type=int ,default=20, help='save summary every n iterations')
    parser.add_argument('--log_every', '-l', type=int, default=20, help='log every l iterations')

    parser.add_argument('--log',action='store_true', help='write to logfile instead of stdout')
    parser.add_argument('--err',action='store_true', help='write to errfile instead of stderr')

    parser.add_argument('--nodownload', action='store_true', help='write files to local downloadir data/<train-test><fold>/')

    parser.add_argument('--datadir', default="data", help='directory to for local data default data/<train-test><fold>/. Will be ignored if nodownload flag is set')

    args = parser.parse_args()

    """ Connection to DB """
    #print os.environ["FIELDDBCONNECTSTRING"]
    #conn = psycopg2.connect(os.environ["FIELDDBCONNECTSTRING"])

    """ GPU management """
    allow_gpu_mem_growth = True
    gpu_memory_fraction = 1
    gpu_id = args.gpu

    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id

    """ Saves """
    max_ckpts_to_keep = args.max_ckpts_to_keep
    ckpt_every_n_hours = args.ckpt_every_n_hours
    save_every = args.save_every # iterations
    run = params2name(args.layers, args.cells, args.dropout, args.fold)
    save_dir = os.path.join(args.savedir,args.model,run)
    init_from = save_dir

    """ redirect stdout stderr """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args.log:
        sys.stdout = open(os.path.join(save_dir,"stdout.log"), 'w')
    if args.err:
        sys.stderr = open(os.path.join(save_dir,"stderr.log"), 'w')


    """ summary logging """
    summary_every = args.summary_every
    print_every = args.log_every

    max_epoch = args.maxepoch
    batch_size = args.batchsize
    keep_prob = args.dropout
    n_layers = args.layers
    n_cell_per_input = args.cells


    # do not change.
    # depends on size of input do not change
    # n_pixels * n_bands + doy = 9 * 6 + 1 = 55
    n_input = 55

    print "Start Training with y_layers {}, n_cell_per_input {}, keep_prob {}, run {}, init_from {}".format(n_layers, n_cell_per_input, keep_prob, run, init_from)

    tf.reset_default_graph()

    tablename = "raster_label_fields"

    if args.nodownload:
        test_localdir = train_localdir = None
    else:
        test_localdir = "data/test".format(args.datadir, args.fold)
        train_localdir = "data/train".format(args.datadir, args.fold)

    test_dataloader = Dataloader(datafolder=test_localdir, batchsize=500)
    train_dataloader = Dataloader(datafolder=train_localdir, batchsize=500)

    n_classes = train_dataloader.nclasses

    """ select network model """

    print("building model graph on device {}".format(gpu_id))
    if args.model in ["lstm","rnn"]:
        model = rnn_model.Model(n_input=n_input, n_classes=n_classes, n_layers=n_layers, batch_size=batch_size,
                                adam_lr=1e-3, dropout_keep_prob=keep_prob, n_cell_per_input=n_cell_per_input, gpu=gpu_id,
                                rnn_cell_type=args.model)
        train = train_rnn

    if args.model == "cnn":
        model = cnn_model.Model(n_input=n_input, n_classes=n_classes, n_layers=n_layers,
                               adam_lr=1e-3, dropout_keep_prob=keep_prob, n_cell_per_input=n_cell_per_input, gpu=gpu_id)
        train = train_cnn


    success = train(model,
                    train_dataloader,
                    test_dataloader,
                    max_epoch=max_epoch,
                    savedir=save_dir,
                    init_from=init_from,
                    log_every=summary_every,
                    save_every=save_every,
                    print_every=print_every,
                    max_ckpts_to_keep=max_ckpts_to_keep,
                    ckpt_every_n_hours=ckpt_every_n_hours,
                    gpu_memory_fraction=gpu_memory_fraction,
                    allow_gpu_mem_growth=allow_gpu_mem_growth)

    if success:
        print "Process terminated successfully"

if __name__ == '__main__':
    main()
