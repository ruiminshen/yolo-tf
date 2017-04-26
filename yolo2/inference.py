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
import tensorflow.contrib.slim as slim
from yolo.inference import leaky_relu


def tiny(net, classes, num_anchors, training=False):
    scope = __name__.split('.')[0] + '_' + inspect.stack()[0][3]
    with slim.arg_scope([slim.layers.conv2d], kernel_size=[3, 3], normalizer_fn=slim.batch_norm, normalizer_params={'center': True, 'scale': True, 'is_training': training, 'updates_collections': None}, activation_fn=leaky_relu), slim.arg_scope([slim.layers.max_pool2d], kernel_size=[2, 2], padding='SAME'):
        index = 0
        channels = 16
        for _ in range(5):
            net = slim.layers.conv2d(net, channels, scope='%s/conv%d' % (scope, index))
            net = slim.layers.max_pool2d(net, scope='%s/max_pool%d' % (scope, index))
            index += 1
            channels *= 2
        net = slim.layers.conv2d(net, 512, scope='%s/conv%d' % (scope, index))
        net = slim.layers.max_pool2d(net, stride=1, scope='%s/max_pool%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, 1024, scope='%s/conv%d' % (scope, index))
        index += 1
        net = slim.layers.conv2d(net, 1024, scope='%s/conv%d' % (scope, index))
    index += 1
    net = slim.layers.conv2d(net, num_anchors * (5 + classes), kernel_size=[1, 1], activation_fn=None, scope='%s/conv' % scope)
    return scope, net
