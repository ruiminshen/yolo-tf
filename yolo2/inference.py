"""
Copyright (C) 2017, 申瑞珉 (Ruimin Shen)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import inspect
import tensorflow as tf
import tensorflow.contrib.slim as slim
from yolo.function import leaky_relu
from yolo2.function import reorg


def tiny(net, classes, num_anchors, training=False):
    scope = __name__.split('.')[0] + '_' + inspect.stack()[0][3]
    net = tf.identity(net, name='%s/input' % scope)
    with slim.arg_scope([slim.layers.conv2d], kernel_size=[3, 3], weights_initializer=tf.truncated_normal_initializer(stddev=0.1), normalizer_fn=slim.batch_norm, normalizer_params={'scale': True}, activation_fn=leaky_relu), slim.arg_scope([slim.layers.max_pool2d], kernel_size=[2, 2], padding='SAME'):
        index = 0
        channels = 16
        for _ in range(5):
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
            index += 1
            channels *= 2
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, stride=1, scope='%s/max_pool%d' % (scope, index))
        index += 1
        channels *= 2
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
    net = slim.layers.conv2d(net, num_anchors * (5 + classes), kernel_size=[1, 1], activation_fn=None, scope='%s/conv' % scope)
    net = tf.identity(net, name='%s/output' % scope)
    return scope, net

TINY_DOWNSAMPLING = (2 ** 5, 2 ** 5)


def darknet(net, classes, num_anchors, training=False, center=False):
    def batch_norm(net):
        net = slim.batch_norm(net, center=center, scale=True, epsilon=1e-5, is_training=training)
        if not center:
            net = tf.nn.bias_add(net, slim.variable('biases', shape=[net.get_shape()[-1]], initializer=tf.zeros_initializer()))
        return net
    scope = __name__.split('.')[0] + '_' + inspect.stack()[0][3]
    net = tf.identity(net, name='%s/input' % scope)
    with slim.arg_scope([slim.layers.conv2d], kernel_size=[3, 3], weights_initializer=tf.truncated_normal_initializer(stddev=0.1), normalizer_fn=batch_norm, activation_fn=leaky_relu), slim.arg_scope([slim.layers.max_pool2d], kernel_size=[2, 2], padding='SAME'):
        index = 0
        channels = 32
        for _ in range(2):
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
            index += 1
            channels *= 2
        for _ in range(2):
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            index += 1
            net = slim.layers.conv2d(net, channels / 2, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
            index += 1
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
            index += 1
            channels *= 2
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels / 2, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels / 2, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        passthrough = tf.identity(net, name=scope + '/passthrough')
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1
        channels *= 2
        # downsampling finished
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels / 2, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels / 2, kernel_size=[1, 1], scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        with tf.name_scope(scope):
            _net = reorg(passthrough)
        net = tf.concat([_net, net], 3, name='%s/concat%d' % (scope, index))
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
    net = slim.layers.conv2d(net, num_anchors * (5 + classes), kernel_size=[1, 1], activation_fn=None, scope='%s/conv' % scope)
    net = tf.identity(net, name='%s/output' % scope)
    return scope, net

DARKNET_DOWNSAMPLING = (2 ** 5, 2 ** 5)


def darknet_tiny(net, classes, num_anchors, training=False, center=False):
    def batch_norm(net):
        net = slim.batch_norm(net, center=center, scale=True, epsilon=1e-5, is_training=training)
        if not center:
            net = tf.nn.bias_add(net, slim.variable('biases', shape=[net.get_shape()[-1]], initializer=tf.zeros_initializer()))
        return net
    scope = __name__.split('.')[0] + '_' + inspect.stack()[0][3]
    net = tf.identity(net, name='%s/input' % scope)
    with slim.arg_scope([slim.layers.conv2d], kernel_size=[3, 3], weights_initializer=tf.truncated_normal_initializer(stddev=0.1), normalizer_fn=batch_norm, activation_fn=leaky_relu), slim.arg_scope([slim.layers.max_pool2d], kernel_size=[2, 2], padding='SAME'):
        index = 0
        channels = 16
        for _ in range(5):
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
            index += 1
            channels *= 2
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, stride=1, scope='%s/max_pool%d' % (scope, index))
        index += 1
        channels *= 2
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
    net = slim.layers.conv2d(net, num_anchors * (5 + classes), kernel_size=[1, 1], activation_fn=None, scope='%s/conv' % scope)
    net = tf.identity(net, name='%s/output' % scope)
    return scope, net

DARKNET_TINY_DOWNSAMPLING = (2 ** 5, 2 ** 5)
