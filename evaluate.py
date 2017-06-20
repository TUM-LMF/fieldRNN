import tensorflow as tf
import cPickle as pickle
import rnn_model
import cnn_model
from DataLoader import DataLoader
import psycopg2
import os
import datetime
import numpy as np
import argparse
from cnn_model import unroll

def main():
    parser = argparse.ArgumentParser(description='Evaluate .')

    parser.add_argument('rundir', type=str, help='directory of tf checkpoint file')
    parser.add_argument('--model', type=str, help="Neural network architecture. 'lstm', 'rnn' or 'cnn' (default lstm)", default='lstm')
    parser.add_argument('--tablename', type=str, help="Database batch table name (default raster_label_fields)", default='raster_label_fields')
    parser.add_argument('--sqlwhere', type=str, help="SQL to select which POIs to load (default: where is_evaluate=True)", default='where is_evaluate=True')
    parser.add_argument('--batchsize', type=int, help="batchsize (500)", default=500)
    parser.add_argument('--gpu', type=int, help="Select gpu (e.g. 0), via environment variable CUDA_VISIBLE_DEVICES (default None)", default=None)

    args = parser.parse_args()

    """ DEBUG """
    #args.rundir = "save/tmp/lstm/4l4r50d1f"
    #args.tablename = "raster_label_fields"
    #args.sqlwhere = "where is_validate=True"

    """ Connection to DB """
    print os.environ["FIELDDBCONNECTSTRING"]
    conn = psycopg2.connect(os.environ["FIELDDBCONNECTSTRING"])

    """ GPU management """
    allow_gpu_mem_growth = True
    gpu_memory_fraction = 1
    gpu_id = args.gpu

    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    dataloader = DataLoader(conn=conn, batch_size=args.batchsize, sql_where=args.sqlwhere,
                            debug=False,
                            do_shuffle=False, do_init_shuffle=True, tablename=args.tablename)

    """
    Load
    parameters
    from init_from model
    """
    with open(os.path.join(args.rundir, "args.pkl"), "rb") as f:
        modelargs = pickle.load(f)

        """
    Create
    new
    model
    object
    with same parameter """
    print("building model graph")

    if args.model in ["rnn","lstm"]:
        model = rnn_model.Model(n_input=modelargs["n_input"], n_classes=modelargs["n_classes"], n_layers=modelargs["n_layers"], batch_size=args.batchsize,
                                adam_lr=modelargs["adam_lr"],rnn_cell_type=args.model , dropout_keep_prob=modelargs["dropout_keep_prob"], n_cell_per_input=modelargs["n_cell_per_input"], gpu=0)
        evaluate=evaluate_rnn

    if args.model == "cnn":
        model = cnn_model.Model(n_input=modelargs["n_input"], n_classes=modelargs["n_classes"], n_layers=modelargs["n_layers"],
                               adam_lr=1e-3, dropout_keep_prob=modelargs["dropout_keep_prob"], n_cell_per_input=modelargs["n_cell_per_input"], gpu=gpu_id)
        evaluate = evaluate_cnn

    probabilities, targets, observations = evaluate(model,dataloader,
                                                    init_dir=args.rundir,
                                                    print_every=20,
                                                    gpu_memory_fraction=gpu_memory_fraction,
                                                    allow_gpu_mem_growth=allow_gpu_mem_growth)

    #np.save(os.path.join(args.rundir, "eval_confusion_matrix.npy"), confusion_matrix)
    np.save(os.path.join(args.rundir, "eval_probabilities.npy"), probabilities)
    np.save(os.path.join(args.rundir, "eval_targets.npy"), targets)
    np.save(os.path.join(args.rundir, "eval_observations.npy"), observations)

def evaluate_rnn(model,
             dataloader,
             print_every=5,
             init_dir=None,
             allow_gpu_mem_growth=True,
             gpu_memory_fraction=0.3):
    """
    This function initialized a model from the <init_from> directory and calculates
    probabilities, and confusion matrices based on all data stored in
    one epoch of dataloader (usually test data)


    :param model:                   rnn_model object containing tensorflow graph
    :param dataloader:              DataLoader object for loading batches
    :param print_every:             console log frequency
    :param allow_gpu_mem_growth:    dynamic growth of gpu vram
    :param gpu_memory_fraction:     hard upper limit for gpu vram

    :returns confusion_matrix       <float> [n_classes x n_classes] rows as targets cols as predicted
    :returns probabilities          <float> [all observations x n_classes] probabilities for each class per observation
    :returns targets                <bool>  [all observations x n_classes] reference data for each class per observation
    :returns observations           <int>   [all_observations]position of observation in the sequence
                                    e.g. [1,2,3,4,1,2,3,4,5,6,1,2,3,4, ...]
    """

    saver = tf.train.Saver()

    # container for output data
    total_cm = np.zeros((model.n_classes, model.n_classes))
    all_scores = np.array([])
    all_targets = np.array([])
    all_obs = np.array([])

    step = 0
    t_last = datetime.datetime.now()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = allow_gpu_mem_growth
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    config.allow_soft_placement = True

    print("start")
    with tf.Session(config=config) as sess:

        sess.run([model.init_op])
        if init_dir is not None:
            if os.path.exists(init_dir):
                ckpt = tf.train.get_checkpoint_state(init_dir)
                print("restoring model from %s" % ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(1, dataloader.num_batches):

            # step as number of features -> invariant to changes in batch size
            step += dataloader.batch_size

            s_db = datetime.datetime.now()
            X, y, seq_lengths = dataloader.next_batch()
            e_db = datetime.datetime.now()

            feed = {model.X: X, model.y_: y, model.seq_lengths: seq_lengths}

            cm, scores, targets, obs = sess.run([model.confusion_matrix, model.scores, model.targets, model.obs],
                                                feed_dict=feed)

            all_obs = np.append(all_obs, obs)
            all_scores = np.append(all_scores, scores)
            all_targets = np.append(all_targets, targets)
            total_cm += cm

            e_tr = datetime.datetime.now()

            dt_db = e_db - s_db
            dt_tr = e_tr - e_db

            field_per_s = dataloader.batch_size / (datetime.datetime.now() - t_last).total_seconds()
            # approximate calculation time
            approx_calc_time = (((dataloader.num_feat) - step) / field_per_s)
            eta = datetime.datetime.now() + datetime.timedelta(seconds=approx_calc_time)

            t_last = datetime.datetime.now()

            if i % print_every == 0:
                cross_entropy = sess.run(model.cross_entropy, feed_dict=feed)
                msg = "Gathering: Iteration {}, feature {}, epoch {}, batch {}/{}: xentr {:.2f} " \
                      "(time: db {}ms; eval {}ms, {} feat/s, eta: {})".format(
                    i,
                    step,
                    dataloader.epoch,
                    dataloader.batch,
                    dataloader.num_batches,
                    cross_entropy,
                    int(dt_db.total_seconds() * 1000),
                    int(dt_tr.total_seconds() * 1000),
                    int(field_per_s),
                    eta.strftime("%d.%b %H:%M")
                )
                print(msg)

    return all_scores.reshape(-1, model.n_classes), \
           all_targets.reshape(-1, model.n_classes).astype(bool), \
           all_obs

def evaluate_cnn(model,
             dataloader,
             print_every=5,
             init_dir=None,
             allow_gpu_mem_growth=True,
             gpu_memory_fraction=0.3):
    """
    This function initialized a model from the <init_from> directory and calculates
    probabilities, and confusion matrices based on all data stored in
    one epoch of dataloader (usually test data)


    :param model:                   rnn_model object containing tensorflow graph
    :param dataloader:              DataLoader object for loading batches
    :param print_every:             console log frequency
    :param allow_gpu_mem_growth:    dynamic growth of gpu vram
    :param gpu_memory_fraction:     hard upper limit for gpu vram

    :returns confusion_matrix       <float> [n_classes x n_classes] rows as targets cols as predicted
    :returns probabilities          <float> [all observations x n_classes] probabilities for each class per observation
    :returns targets                <bool>  [all observations x n_classes] reference data for each class per observation
    :returns observations           <int>   [all_observations]position of observation in the sequence
                                    e.g. [1,2,3,4,1,2,3,4,5,6,1,2,3,4, ...]
    """

    saver = tf.train.Saver()

    # container for output data
    total_cm = np.zeros((model.n_classes, model.n_classes))
    all_scores = np.array([])
    all_targets = np.array([])
    all_obs = np.array([])

    step = 0
    t_last = datetime.datetime.now()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = allow_gpu_mem_growth
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    config.allow_soft_placement = True

    print("start")
    with tf.Session(config=config) as sess:

        sess.run([model.init_op])
        if init_dir is not None:
            if os.path.exists(init_dir):
                ckpt = tf.train.get_checkpoint_state(init_dir)
                print("restoring model from %s" % ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)

                with open(init_dir + "/steps.txt", "r") as f:
                    line = f.read()
                    step_, epoch_ = line.split(" ")
                    step = int(step_)
                    dataloader.epoch = int(epoch_)

        for i in range(1, dataloader.num_batches):

            # step as number of features -> invariant to changes in batch size
            step += dataloader.batch_size

            s_db = datetime.datetime.now()
            X, y, seq_lengths = dataloader.next_batch()
            e_db = datetime.datetime.now()

            # unroll also index of observation. -> TODO integrate in unroll function, but need to update also dependencies
            batch_size, max_seqlengths, n_input = X.shape
            ones = np.ones([batch_size, max_seqlengths])
            mask_ = np.arange(0, max_seqlengths) * ones < (seq_lengths * ones.T).T
            mask = mask_.reshape(-1)
            obs_ = np.arange(0, max_seqlengths) * ones
            obs = obs_.reshape(-1)[mask]

            """ unroll data """
            X, y = unroll(X, y, seq_lengths)

            feed = {model.X: X, model.y: y, model.batch_size: X.shape[0]}

            scores, targets = sess.run([model.scores, model.targets],
                                                feed_dict=feed)

            all_scores = np.append(all_scores, scores)
            all_targets = np.append(all_targets, targets)

            e_tr = datetime.datetime.now()

            dt_db = e_db - s_db
            dt_tr = e_tr - e_db

            field_per_s = dataloader.batch_size / (datetime.datetime.now() - t_last).total_seconds()
            # approximate calculation time
            approx_calc_time = (((dataloader.num_feat) - step) / field_per_s)
            eta = datetime.datetime.now() + datetime.timedelta(seconds=approx_calc_time)

            t_last = datetime.datetime.now()

            if i % print_every == 0:
                cross_entropy = sess.run(model.cross_entropy, feed_dict=feed)
                msg = "Gathering: Iteration {}, feature {}, epoch {}, batch {}/{}: xentr {:.2f} " \
                      "(time: db {}ms; eval {}ms, {} feat/s, eta: {})".format(
                    i,
                    step,
                    dataloader.epoch,
                    dataloader.batch,
                    dataloader.num_batches,
                    cross_entropy,
                    int(dt_db.total_seconds() * 1000),
                    int(dt_tr.total_seconds() * 1000),
                    int(field_per_s),
                    eta.strftime("%d.%b %H:%M")
                )
                print(msg)

    return all_scores.reshape(-1, model.n_classes), \
           all_targets.reshape(-1, model.n_classes).astype(bool), \
           obs

if __name__ == '__main__':
    main()