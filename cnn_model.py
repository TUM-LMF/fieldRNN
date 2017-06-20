import tensorflow as tf
from tensorflow.contrib import rnn as rnn_cell
import numpy as np
import io
from util.tf_utils import tf_confusion_metrics
import inspect
import util.eval as eval


class Model:
    """
    -- Copied from RNN TODO update to FC --

    Tensorflow graph using ful

    Tensorflow Graph using Fully connected layers and fully connected softmax layer for field identification
    with multispectral/temporal data acquired from satellite imagery

    Params
        tf placeholders:
            X           Input data cube of dimensions [batch_size x max_observations x n_input]
            y           Target data Tensor of dimensions [batch_size x max_observations]
            seq_lenghts Number of observations for each batch if observation < max_obs data is
                        padded with zeros [batch_size]

        input parameters:
            n_input     length of observed pixel values. [n_pixels * n_bands + n_time]
                n_pixels    number of observed pixels (default 3*3)
                n_bands     number of observed bands  (default 6)
                n_time      number of time parameters (default 1 e.g. day of year)

            n_classes   number of target classes
            batch_size  number of batches
            max_obs     maximum number of observations if seq_lengs < max_obs matrices will be padded
                        controls number of iterations in rnn layers (aka sequence length)


        network specific parameters
            n_layers    number of rnn layers (aka depth)
            learning_rate
            dropout_keep_prob
            logdir

    Marc.Russwurm@tum.de
    """

    def __init__(self, n_input=9 * 6 + 1, n_classes=20,
                 n_layers=2, dropout_keep_prob=.5, adam_lr=1e-3, adam_b1=0.9, adam_b2=0.999, adam_eps=1e-8,
                 fc_w_stddev=0.1, fc_b_offset=0.1, n_cell_per_input=1, activation_func=None, gpu=None):
        # save input arguments
        self.args = inspect.getargvalues(inspect.currentframe()).locals
        del self.args["self"]  # delete self

        self.n_classes = n_classes

        if activation_func is None:
            activation_func = tf.nn.sigmoid
        # alternative tf.nn.relu

        # take
        self.n_neurons = n_neurons = n_cell_per_input * n_input

        with tf.device(None):

            with tf.variable_scope('input'):
                # block of [batch_size x max_obs x n_input]
                self.X = X = tf.placeholder(tf.float32, [None, n_input], name="X")
                self.y = y = tf.placeholder(tf.float32, [None, n_classes], name="y")
                self.batch_size = batch_size = tf.placeholder(tf.int32, name="batch_size")
            with tf.name_scope('FC'):


                # first fc layer: expand neuron dimensions from n_input to n_neurons
                # list of fully connected weights matrices

                fc_in = X

                # first fc layer X:(batchsize x n_input) -> fc_in (batchsize x n_neurons)
                fc_W0 = tf.Variable(tf.truncated_normal([n_input, n_neurons], stddev=fc_w_stddev), name="W0")
                fc_b0 = tf.Variable(tf.constant(fc_b_offset, shape=[n_neurons]), name="b0")
                h = activation_func(tf.matmul(fc_in, fc_W0) + fc_b0)

                h = tf.nn.dropout(h, dropout_keep_prob)

                # for all other fc layers
                fc_W = []
                fc_b = []
                for i in range(1, n_layers):
                    W = tf.Variable(tf.truncated_normal([n_neurons, n_neurons], stddev=fc_w_stddev), name="W" + str(i))
                    b = tf.Variable(tf.constant(fc_b_offset, shape=[n_neurons]), name="b" + str(i))

                    h = tf.matmul(h, W) + b

                    # apply activation function
                    h = activation_func(h)

                    h = tf.nn.dropout(h, dropout_keep_prob)

                fc_out = h

            with tf.name_scope('fc_softmax'):
                # reshape outputs to: block of [batch_size * max_obs x rnn_size]
                softmax_in = tf.reshape(fc_out, [-1, n_neurons])
                softmax_w = tf.Variable(tf.truncated_normal([n_neurons, n_classes], stddev=fc_w_stddev), name="W_softmax")
                softmax_b = tf.Variable(tf.constant(fc_b_offset, shape=[n_classes]), name="b_softmax")

                self.logits = logits = tf.matmul(softmax_in, softmax_w) + softmax_b

            with tf.name_scope('train'):
                # Define loss and optimizer

                # create mask for cross entropies incases where seq_lengths < max_max_obs
                # masking from http://stackoverflow.com/questions/34128104/tensorflow-creating-mask-of-varied-lengths

                """ no masking needed
                with tf.name_scope('mask'):
                    lengths_transposed = tf.expand_dims(seq_lengths, 1)

                    range = tf.range(0, max_obs, 1)
                    range_row = tf.expand_dims(range, 0)

                    self.mask = mask = tf.less(range_row, lengths_transposed)
                """

                self.cross_entropy_matrix = cross_entropy_matrix = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

                # normalize with total number of observations
                self.cross_entropy = cross_entropy = tf.reduce_sum(cross_entropy_matrix) / tf.cast(batch_size,"float32")

                tf.summary.scalar('cross_entropy', cross_entropy)
                # grad_train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
                self.train_op = tf.train.AdamOptimizer(learning_rate=adam_lr, beta1=adam_b1, beta2=adam_b2,
                                                       epsilon=adam_eps).minimize(cross_entropy)
                # tf.summary.scalar('learning_rate', learning_rate)

            with tf.name_scope('evaluation'):

                self.probabilities = probs = tf.nn.softmax(logits, name="full_probability_matrix")

                # Evaluate model
                predicted = tf.argmax(logits, 1)
                targets = tf.argmax(y, 1)

                correct_pred = tf.equal(predicted, targets)
                self.accuracy_op = accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32)) / tf.cast(batch_size, tf.float32)
                tf.summary.scalar('accuracy', accuracy)

                self.probs_list = probs_list = tf.reshape(probs, (-1, n_classes))
                predicted_list = tf.reshape(predicted, [-1])
                targets_list = tf.reshape(targets, [-1])

                one_hot_targets = tf.one_hot(targets_list, n_classes)
                scores = tf.boolean_mask(probs_list, tf.cast(one_hot_targets, tf.bool))

                self.scores = probs_list
                self.targets = tf.reshape(y, [-1,n_classes])

                # drop all values which are > seqlength
                #self.scores = tf.boolean_mask(scores, mask_list)
                #self.targets = tf.boolean_mask(targets_list, mask_list)
                #self.obs = tf.boolean_mask(obs_list, mask_list)


                """
                self.confusion_matrix = confusion_matrix = tf.contrib.metrics.confusion_matrix(
                    tf.boolean_mask(targets_list, mask_list),
                    tf.boolean_mask(predicted_list, mask_list),
                    num_classes=n_classes)



                confusion_matrix = tf.cast(confusion_matrix, tf.uint8)
                confusion_matrix = tf.expand_dims(confusion_matrix, 2)
                confusion_matrix = tf.expand_dims(confusion_matrix, 0)
                tf.summary.image("confusion matrix", confusion_matrix, max_outputs=3)

                logits_ = tf.cast(logits, tf.uint8)
                logits_ = tf.expand_dims(logits_, 3)
                tf.summary.image("logits", logits_, max_outputs=1)

                probs_ = tf.cast(probs*255, tf.uint8)
                probs_ = tf.expand_dims(probs_, 3)
                tf.summary.image("probabilities", probs_, max_outputs=1)

                targets_ = tf.cast(y_, tf.uint8)
                targets_ = tf.expand_dims(targets_, 3)
                tf.summary.image("targets", targets_, max_outputs=1)

                # tf.add_to_collection(tf.GraphKeys.SUMMARIES, cm_im_summary)
                 """

            # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
            self.merge_summary_op = tf.summary.merge_all()
            self.init_op = tf.global_variables_initializer()


def main():
    # model = Model()
    test()

def unroll(x, y, seq_lengths):
    """
        Reshapes and masks input and output data from

        X(batchsize x n_max_obs x n_input) -> X_ (new_batchsize x n_input)
        y(batchsize x n_max_obs x n_classes) -> X_ (new_batchsize x n_classes)

        new_batch_size is variable representing batchsize * n_max_obs - invalid_observations
        with invalid observations being observations > seq_length -> means
        if at one point only 24 of maximum 26 images are available X is usually padded with zeros
        this masking removes the last two observations

    :return:
    """
    # create mask for valid times of acquisition
    batch_size, max_seqlengths, n_input = x.shape
    np.arange(0, max_seqlengths) * np.ones((batch_size, max_seqlengths))
    ones = np.ones([batch_size, max_seqlengths])
    mask = np.arange(0, max_seqlengths) * ones < (seq_lengths * ones.T).T

    new_x = x[mask]
    new_y = y[mask]

    return new_x, new_y

def test():
    import os
    import pickle

    n_input = 9 * 6 + 1
    n_classes = 20
    batch_size = 50
    max_obs = 26
    n_classes = 38

    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

    model = Model(n_input=n_input, n_classes=n_classes, n_layers=2, batch_size=batch_size,
                  adam_lr=1e-3, dropout_keep_prob=0.5, n_cell_per_input=4)

    savedir = "tmp"
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # dump pickle args for loading
    #pickle.dump(model.args, open(os.path.join(savedir, "args.pkl"), "wb"))
    # dump human readable args
    #open(os.path.join(savedir, "args.txt"), "w").write(str(model.args))

    init_from = None
    if init_from is not None:
        args = pickle.load(open(os.path.join(init_from, "args.pkl"), "rb"))


    X = np.random.rand(batch_size, max_obs, n_input)
    y = np.random.rand(batch_size, max_obs, n_classes)



    seq_length = np.random.randint(16, max_obs, batch_size)

    with tf.Session() as sess:
        sess.run([model.init_op])

        feed = {model.X: X, model.y: y}
        # training step
        for i in range(1, 30):
            train_op, cross_entropy = \
                sess.run([model.train_op,
                          model.cross_entropy], feed_dict=feed)

        print("done")


if __name__ == '__main__':
    main()

