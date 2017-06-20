import tensorflow as tf
from tensorflow.contrib import rnn as rnn_cell
import numpy as np
import io
from util.tf_utils import tf_confusion_metrics
import inspect
import util.eval as eval


class Model():
    """
    Tensorflow Graph using Recurrent LSTM layers and fully connected softmax layer for field identification
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

    def __init__(self, n_input=9 * 6 + 1, n_classes=20, batch_size=50, max_obs=26,
                 n_layers=2, dropout_keep_prob=.5, adam_lr=1e-3, adam_b1=0.9, adam_b2=0.999, adam_eps=1e-8,
                 fc_w_stddev=0.1, fc_b_offset=0.1, n_cell_per_input=1,rnn_cell_type="basiclstm", gpu=None):
        # save input arguments
        self.args = inspect.getargvalues(inspect.currentframe()).locals
        del self.args["self"]  # delete self argument

        self.n_classes = n_classes

        with tf.device(None):

            with tf.variable_scope('input'):
                # block of [batch_size x max_obs x n_input]
                self.X = tf.placeholder(tf.float32, [batch_size, max_obs, n_input], name="X")
                self.y_ = self.y = y_ = tf.placeholder(tf.float32, [batch_size, max_obs, n_classes], name="y")
                self.seq_lengths = seq_lengths = tf.placeholder(tf.int32, [batch_size], name="seq_lengths")
                #self.y = y = tf.reshape(self.y_, [-1, n_classes], name="y")

            with tf.name_scope('RNN'):
                self.n_rnn_cells = n_rnn_cells = n_cell_per_input * n_input

                if rnn_cell_type == "basiclstm":
                    cell = rnn_cell.BasicLSTMCell(n_rnn_cells)
                if rnn_cell_type == "lstm":
                    cell = rnn_cell.LSTMCell(n_rnn_cells)
                if rnn_cell_type == "lstm_peephole":
                    cell = rnn_cell.LSTMCell(n_rnn_cells, use_peepholes=True)
                elif rnn_cell_type == "gru":
                    cell = rnn_cell.BasicLSTMCell(n_rnn_cells)
                elif rnn_cell_type == "rnn":
                    cell = rnn_cell.BasicRNNCell(n_rnn_cells)

                # dropout Wrapper
                cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=dropout_keep_prob)
                self.cell = cell = rnn_cell.MultiRNNCell([cell] * n_layers)

                # tensor with class labels of dimension [batch_size x max_obs]
                # defined as Variable to carry values to next iteration (not trainable must be declared explicitly)
                self.state = state = cell.zero_state(batch_size, tf.float32)

                # rnn_outputs: block of [batch_size x max_obs x rnn_size]
                # data is padded with zeros after seq_length
                outputs, last_states = tf.nn.dynamic_rnn(cell, self.X, initial_state=state, sequence_length=seq_lengths,
                                                         time_major=False)

                self.outputs = outputs
                self.last_states = last_states

            with tf.name_scope('fc'):
                # reshape outputs to: block of [batch_size * max_obs x rnn_size]
                softmax_in = tf.reshape(outputs, [-1, n_rnn_cells])
                softmax_w = tf.Variable(tf.truncated_normal([n_rnn_cells, n_classes], stddev=fc_w_stddev), name="W_softmax")
                softmax_b = tf.Variable(tf.constant(fc_b_offset, shape=[n_classes]), name="b_softmax")

                softmax_out = tf.matmul(softmax_in, softmax_w) + softmax_b
                self.logits = logits = tf.reshape(softmax_out, [batch_size, -1, n_classes])

            with tf.name_scope('train'):
                # Define loss and optimizer

                # create mask for cross entropies incases where seq_lengths < max_max_obs
                # masking from http://stackoverflow.com/questions/34128104/tensorflow-creating-mask-of-varied-lengths

                with tf.name_scope('mask'):
                    lengths_transposed = tf.expand_dims(seq_lengths, 1)

                    range = tf.range(0, max_obs, 1)
                    range_row = tf.expand_dims(range, 0)

                    self.mask = mask = tf.less(range_row, lengths_transposed)

                self.cross_entropy_matrix = cross_entropy_matrix = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_)
                self.masked_cross_entropy_matrix = masked_cross_entropy_matrix = tf.where(mask, cross_entropy_matrix,
                                                                                           tf.zeros(mask.get_shape()))
                self.cross_entropy_matrix = cross_entropy_matrix = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_)

                # normalize with total number of observations
                self.cross_entropy = cross_entropy = tf.reduce_sum(cross_entropy_matrix) / tf.cast(
                    tf.reduce_sum(seq_lengths), tf.float32)
                tf.summary.scalar('cross_entropy', cross_entropy)
                # grad_train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
                self.train_op = tf.train.AdamOptimizer(learning_rate=adam_lr, beta1=adam_b1, beta2=adam_b2,
                                                       epsilon=adam_eps).minimize(cross_entropy)
                # tf.summary.scalar('learning_rate', learning_rate)

            with tf.name_scope('evaluation'):

                self.probabilities = probs = tf.nn.softmax(logits, name="full_probability_matrix")

                # Evaluate model
                predicted = tf.argmax(logits, 2)
                targets = tf.argmax(y_, 2)

                correct_pred = tf.equal(predicted, targets)
                masked_correct_pred = tf.logical_and(mask, correct_pred)
                self.accuracy_op = accuracy = tf.reduce_sum(tf.cast(masked_correct_pred, tf.float32)) / tf.cast(
                    tf.reduce_sum(seq_lengths), tf.float32)
                tf.summary.scalar('accuracy', accuracy)

                self.probs_list = probs_list = tf.reshape(probs, (-1, n_classes))
                predicted_list = tf.reshape(predicted, [-1])
                targets_list = tf.reshape(targets, [-1])

                mask_list = tf.reshape(mask, [-1])

                one_hot_targets = tf.one_hot(targets_list, n_classes)
                scores = tf.boolean_mask(probs_list, tf.cast(one_hot_targets, tf.bool))

                # mask of individual number of observations
                obs_list = tf.tile(tf.range(0, max_obs), [batch_size])
                obs_matrix = tf.matmul(tf.expand_dims(obs_list, 1), tf.ones([1, n_classes], dtype=tf.int32))



                probs_matrix_mask = probs_matrix_mask = tf.transpose(tf.reshape(tf.tile(mask_list, [n_classes]),[n_classes,-1]))

                self.scores = tf.boolean_mask(probs_list, probs_matrix_mask)
                self.targets = tf.boolean_mask(tf.reshape(y_, [-1,n_classes]), probs_matrix_mask)
                self.obs = tf.boolean_mask(obs_list, mask_list)

                # drop all values which are > seqlength
                #self.scores = tf.boolean_mask(scores, mask_list)
                #self.targets = tf.boolean_mask(targets_list, mask_list)
                #self.obs = tf.boolean_mask(obs_list, mask_list)



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

            # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
            self.merge_summary_op = tf.summary.merge_all()
            self.init_op = tf.global_variables_initializer()


def main():
    # model = Model()
    test()


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
    pickle.dump(model.args, open(os.path.join(savedir, "args.pkl"), "wb"))
    # dump human readable args
    open(os.path.join(savedir, "args.txt"), "w").write(str(model.args))

    init_from = "tmp"
    if init_from is not None:
        args = pickle.load(open(os.path.join(init_from, "args.pkl"), "rb"))



    X = np.random.rand(batch_size, max_obs, n_input)
    y = np.random.rand(batch_size, max_obs, n_classes)

    seq_length = np.random.randint(16, max_obs, batch_size)

    summaryWriter = tf.summary.FileWriter("tensorboard/test", graph=tf.get_default_graph())

    with tf.Session() as sess:
        sess.run([model.init_op])

        feed = {model.X: X, model.y_: y, model.seq_lengths: seq_length}
        # training step
        for i in range(1, 30):
            train_op, cross_entropy, new_confusion_matrix = \
                sess.run([model.train_op,
                          model.cross_entropy,
                          model.confusion_matrix], feed_dict=feed)

            confusion_matrix += new_confusion_matrix
            print(cross_entropy)

            a,b = eval.class_evaluation(confusion_matrix)

            scores, targets = sess.run([model.scores, tf.reshape(model.targets, [-1])], feed_dict=feed)
            fpr, tpr, threshold = roc_curve(targets, scores, 0)

            summary = sess.run(model.merge_summary_op, feed_dict=feed)
            summaryWriter.add_summary(summary, i)

        #buf = plots.plot_confusion_matrix(confusion_matrix, range(1, n_classes))
        #image = tf.image.decode_png(buf.getvalue(), channels=4)
        #image = tf.expand_dims(image, 0)
        #summary_op = tf.image_summary("matplotlib conf matrix", image)
        #summary = sess.run(summary_op)
       # summaryWriter.add_summary(summary, i)

        print("done")


if __name__ == '__main__':
    main()

