from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def weight_variable(shape, name):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, receptive_field, channels, name):
    kernel_shape = receptive_field + channels
    bias_shape = [channels[-1]]

    W = weight_variable(kernel_shape, name+'-W')
    b = bias_variable(bias_shape, name+'-b')

    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_bias = tf.nn.bias_add(conv, b)

    return tf.nn.relu(conv_bias)

def inference(input, batch_size, is_training=True):

    # 3x128x128
    print("input ", input.shape) # conv 5x5
    with tf.name_scope('conv1'):
        h_conv1 = conv2d(input, [5, 5], [3, 16], 'conv1')

        print("h_conv1  ", h_conv1.shape)


    # 16x128x128
    with tf.name_scope('pool1'):  # pooling 3x3
        h_pool1 = tf.nn.max_pool(h_conv1,
                                 ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1], padding='SAME')

        print("h_pool1  ", h_pool1.shape)


    # 16x64x64
    with tf.name_scope('conv2'): # conv 5x5
        h_conv2 = conv2d(h_pool1, [5, 5], [16, 32], 'conv2')

        print("h_conv2  ", h_conv2.shape)


    # 32x64x64
    with tf.name_scope('pool2'):  # pooling 3x3
        h_pool2 = tf.nn.max_pool(h_conv2,
                                 ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1], padding='SAME')
        print("h_pool2  ", h_pool2.shape)


    # 32x32x32
    with tf.name_scope('conv3'): # conv 3x3
        h_conv3 = conv2d(h_pool2, [3, 3], [32, 64], 'conv3')

        print("h_conv3  ", h_conv3.shape)


    # 64x32x32
    with tf.name_scope('pool3'):  # pooling 3x3
        h_pool3 = tf.nn.max_pool(h_conv3,
                                 ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1], padding='SAME')
        print("h_pool3  ", h_pool3.shape)


    # 64x16x16
    with tf.name_scope('conv4'): # conv 3x3
        h_conv4 = conv2d(h_pool3, [3, 3], [64, 128], 'conv4')

        print("h_conv4  ", h_conv4.shape)


    # 128x16x16
    with tf.name_scope('pool4'):  # pooling 3x3
        h_pool4 = tf.nn.max_pool(h_conv4,
                                 ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1], padding='SAME')
        print("h_pool4  ", h_pool4.shape)


    # 128x8x8 = 8192
    with tf.name_scope('fc1'):
        h_pool4_flat = tf.reshape(h_pool4, [batch_size, -1])
        dim = h_pool4_flat.get_shape()[1].value

        W_fc1 = weight_variable([dim, 1024], 'fc1-W')
        b_fc1 = bias_variable([1024], 'fc1-b')

        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
        print("h_fc1    ", h_fc1.shape)

    with tf.name_scope('dropout1'):
        # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=0.8)
        h_fc1_drop = tf.layers.dropout(h_fc1, rate=0.2, training=is_training) # rate=drop rate

    # 1024

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 5], 'fc2-W')
        b_fc2 = bias_variable([5], 'fc2-b')

        h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        print("h_fc2    ", h_fc2.shape)


    return h_fc2
