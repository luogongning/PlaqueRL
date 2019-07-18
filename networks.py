from __future__ import division

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import hdf5storage
import logging

logger = logging.getLogger('networks')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class ADNetwork:
    """
    input : 112 x 112 x 3, RGB

    conv1 : 7x7x96c, 96b
    relu
    lrnorm(5 2 1.0000e-04 0.7500)
    pool : 2x2 pool?

    conv2 : 5x5x256c, 256b
    relu
    lrnorm
    pool

    conv3 : 3x3x512c, 512b
    relu

    fc4(conv) : 3x3x512c, 512b
    relu
    dropout

    concat(+action_history)

    fc5(conv) : 1x1x512c, 512b (input size is 622=512+110)
    relu
    dropout=x16

    fc6_1(conv, from x16, predictions) : 1x1x512x11, zero biased
    fc6_2(conv, from x16, prediction score) : 1x1x512x2, zero biased
    """
    ACTIONS = np.array([
        [-1, 0, 0, 0],
        [-2, 0, 0, 0],
        [+1, 0, 0, 0],
        [+2, 0, 0, 0],
        [0, -1, 0, 0],
        [0, -2, 0, 0],
        [0, +1, 0, 0],
        [0, +2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, -1, -1],
        [0, 0, 1, 1]
        # terminated
    ], dtype=np.float16)
    NUM_ACTIONS = 11
    NUM_ACTION_HISTORY = 10
    ACTION_IDX_STOP = 8

    def __init__(self, learning_rate=1e-04):
        self.input_tensor = None
        self.label_tensor = None
        self.class_tensor = None
        self.action_history_tensor = None
        self.layer_feat = None
        self.layer_actions = None
        self.layer_scores = None

        self.loss_actions = None
        self.loss_cls = None

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.weighted_grads_fc1 = None
        self.weighted_grads_fc2 = None
        self.var_grads_fc1 = None
        self.var_grads_fc2 = None
        self.weighted_grads_op1 = None
        self.weighted_grads_op2 = None

    def read_original_weights(self, tf_session, path='./models/adnet-original/net_rl_weights.mat'):
        """
        original mat file contains
        I converted 'net_rl.mat' file to 'net_rl_weights.mat' saving only weights in v7.3 format.
        """
        init = tf.global_variables_initializer()
        tf_session.run(init)
        logger.info('all global variables initialized')

        weights = hdf5storage.loadmat(path)

        for var in tf.trainable_variables():
            key = var.name.replace('/weights:0', 'f').replace('/biases:0', 'b')

            if key == 'fc6_1b':
                # add 0.01
                # reference : https://github.com/hellbell/ADNet/blob/master/adnet_test.m#L39
                val = np.zeros(var.shape) + 0.01
            elif key == 'fc6_2b':
                # all zeros
                val = np.zeros(var.shape)
            else:
                val = weights[key]

                # need to make same shape.
                val = np.reshape(val, var.shape.as_list())

            tf_session.run(var.assign(val))
            logger.info('%s : original weights assigned. [0]=%s' % (var.name, str(val[0])[:20]))

        print(tf_session.run(tf.report_uninitialized_variables()))

        return weights

    def create_network(self, input_tensor, label_tensor, class_tensor, action_history_tensor, is_training):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        self.class_tensor = class_tensor
        self.action_history_tensor = action_history_tensor

        # feature extractor - convolutions
        net = slim.convolution(input_tensor, 96, [7, 7], 2, padding='VALID', scope='conv1',
                               activation_fn=tf.nn.relu)
        net = tf.nn.lrn(net, depth_radius=5, bias=2, alpha=1e-4*5, beta=0.75)
        net = slim.pool(net, [3, 3], 'MAX', stride=2, padding='VALID', scope='pool1')

        net = slim.convolution(net, 256, [5, 5], 2, padding='VALID', scope='conv2',
                               activation_fn=tf.nn.relu)
        net = tf.nn.lrn(net, depth_radius=5, bias=2, alpha=1e-4*5, beta=0.75)
        net = slim.pool(net, [3, 3], 'MAX', stride=2, padding='VALID', scope='pool2')

        net = slim.convolution(net, 512, [3, 3], 1, padding='VALID', scope='conv3',
                               activation_fn=tf.nn.relu)
        self.layer_feat = net

        # fc layers
        net = slim.convolution(net, 512, [3, 3], 1, padding='VALID', scope='fc4',
                               activation_fn=tf.nn.relu)
        net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout')

        net = tf.concat([net, action_history_tensor], axis=-1)
        net = slim.convolution(net, 512, [1, 1], 1, padding='VALID', scope='fc5',
                               activation_fn=tf.nn.relu)
        net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='dropout_x16')

        # auxilaries
        out_actions = slim.convolution(net, 11, [1, 1], 1, padding='VALID', scope='fc6_1', activation_fn=None)
        out_scores = slim.convolution(net, 2, [1, 1], 1, padding='VALID', scope='fc6_2', activation_fn=None)
        out_actions = flatten_convolution(out_actions)
        out_scores = flatten_convolution(out_scores)
        self.layer_actions = tf.nn.softmax(out_actions)
        self.layer_scores = tf.nn.softmax(out_scores)

        # losses
        self.loss_actions = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_tensor, logits=out_actions)
        self.loss_cls = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=class_tensor, logits=out_scores)

        # finetune ops
        var_fc = [var for var in tf.trainable_variables() if 'fc' in var.name and 'fc6_2' not in var.name]
        self.var_grads_fc1 = var_fc
        gradients1 = tf.gradients(self.loss_actions, xs=var_fc)      # only finetune on fc1 layers
        self.weighted_grads_fc1 = []
        for var, grad in zip(var_fc, gradients1):
            self.weighted_grads_fc1.append(10 * grad)
            continue
            if 'fc6_1/weights' in var.name:
                self.weighted_grads_fc1.append(20 * grad)
            elif 'fc6_1/biases' in var.name:
                self.weighted_grads_fc1.append(40 * grad)
            elif 'weights' in var.name:
                self.weighted_grads_fc1.append(20 * grad)
            elif 'biases' in var.name:
                self.weighted_grads_fc1.append(10 * grad)
            else:
                raise

        var_fc = [var for var in tf.trainable_variables() if 'fc' in var.name and 'fc6_1' not in var.name]
        self.var_grads_fc2 = var_fc
        gradients2 = tf.gradients(self.loss_cls, xs=var_fc)          # only finetune on fc2 layers
        self.weighted_grads_fc2 = []
        for var, grad in zip(var_fc, gradients2):
            self.weighted_grads_fc2.append(10 * grad)
            continue
            if 'weights' in var.name:
                self.weighted_grads_fc2.append(20 * grad)
            elif 'biases' in var.name:
                self.weighted_grads_fc2.append(10 * grad)
            else:
                raise

        self.weighted_grads_op1 = self.optimizer.apply_gradients(zip(self.weighted_grads_fc1, self.var_grads_fc1))
        self.weighted_grads_op2 = self.optimizer.apply_gradients(zip(self.weighted_grads_fc2, self.var_grads_fc2))


def flatten_convolution(tensor_in):
    tendor_in_shape = tensor_in.get_shape()
    tensor_in_flat = tf.reshape(tensor_in, [tendor_in_shape[0].value or -1, np.prod(tendor_in_shape[1:]).value])
    return tensor_in_flat


if __name__ == '__main__':
    input_node = tf.placeholder(tf.float32, shape=(None, 112, 112, 3), name='patch')
    tensor_lb_action = tf.placeholder(tf.int32, shape=(None, ), name='lb_action')    # 11 actions
    tensor_lb_class = tf.placeholder(tf.int32, shape=(None, ), name='lb_class')      # 2 actions
    action_history_tensor = tf.placeholder(tf.float32, shape=(None, 1, 1, ADNetwork.NUM_ACTIONS * ADNetwork.NUM_ACTION_HISTORY), name='action_history')
    is_training = tf.placeholder(tf.bool, name='is_training')

    adnet = ADNetwork()
    adnet.create_network(input_node, tensor_lb_action, tensor_lb_class, action_history_tensor, is_training)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        # load all pretrained weights
        adnet.read_original_weights(sess)

        # zero input
        zeros = np.zeros(shape=(1, 112, 112, 3), dtype=np.float32)
        zeros_out = sess.run(adnet.layer_feat, feed_dict={input_node: zeros})
        pass
