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


def leaky_relu(inputs, alpha=.1):
    with tf.name_scope('leaky_relu'):
        data = tf.identity(inputs, name='data')
        return tf.maximum(data, alpha * data)


def tiny(net, classes, boxes_per_cell, training=False):
    scope = __name__.split('.')[0] + '_' + inspect.stack()[0][3]
    net = tf.identity(net, name='%s/input' % scope)
    with slim.arg_scope([slim.layers.conv2d], kernel_size=[3, 3], activation_fn=leaky_relu), slim.arg_scope([slim.layers.max_pool2d], kernel_size=[2, 2], padding='SAME'):
        index = 0
        net = slim.layers.conv2d(net, 16, scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, 32, scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, 64, scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, 128, scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, 256, scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, 512, scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, 512, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, 1024, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, 256, scope='%s/conv%d' % (scope, index))
    net = tf.identity(net, name='%s/conv' % scope)
    _, cell_height, cell_width, _ = net.get_shape().as_list()
    net = slim.layers.flatten(net, scope='%s/flatten' % scope)
    with slim.arg_scope([slim.layers.fully_connected], activation_fn=leaky_relu), slim.arg_scope([slim.layers.dropout], keep_prob=.5, is_training=training):
        index = 0
        net = slim.layers.fully_connected(net, 256, scope='%s/fc%d' % (scope, index))
        net = slim.layers.dropout(net, scope='%s/dropout%d' % (scope, index))
        index += 1
        net = slim.layers.fully_connected(net, 4096, scope='%s/fc%d' % (scope, index))
        net = slim.layers.dropout(net, scope='%s/dropout%d' % (scope, index))
    net = slim.layers.fully_connected(net, cell_width * cell_height * (classes + boxes_per_cell * 5), activation_fn=None, scope='%s/fc' % scope)
    net = tf.identity(net, name='%s/output' % scope)
    return scope, net
