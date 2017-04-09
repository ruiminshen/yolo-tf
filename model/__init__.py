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

import math
import tensorflow as tf


class ParamConv(list):
    def __init__(self, channels, layers, seed=None):
        self.channels = channels
        for i, (size, kernel1, kernel2) in enumerate(layers[['size', 'kernel1', 'kernel2']].values):
            with tf.variable_scope('conv%d' % i):
                weight = tf.Variable(tf.truncated_normal([kernel1, kernel2, channels, size], stddev=1.0 / math.sqrt(channels * kernel1 * kernel2), seed=seed), name='weight')
                bais = tf.Variable(tf.zeros([size]), name='bais')
                list.append(self, (weight, bais))
                channels = size
    
    def get_size(self, i):
        return list.__getitem__(self, i)[1].get_shape()[0].value


class ParamFC(list):
    def __init__(self, inputs, layers, seed=None):
        for i, size in enumerate(layers['size'].values):
            with tf.variable_scope('fc%d' % i):
                weight = tf.Variable(tf.truncated_normal([inputs, size], stddev=1.0 / math.sqrt(inputs), seed=seed), name='weight')
                bais = tf.Variable(tf.zeros([size]), name='bais')
                list.append(self, (weight, bais))
                inputs = size
        self.seed = seed
    
    def __call__(self, outputs):
        inputs = self.get_size(-1)
        with tf.variable_scope('fc'):
            weight = tf.Variable(tf.truncated_normal([inputs, outputs], stddev=1.0 / math.sqrt(inputs), seed=self.seed), name='weight')
            bais = tf.Variable(tf.zeros([outputs]), name='bais')
            list.append(self, (weight, bais))
    
    def get_size(self, i):
        return list.__getitem__(self, i)[1].get_shape()[0].value


class ModelConv(list):
    def __init__(self, image, param, layers, training=False, seed=None):
        for i, ((weight, bais), (stride1, stride2, pooling1, pooling2, act, norm)) in enumerate(zip(param, layers[['stride1', 'stride2', 'pooling1', 'pooling2', 'act', 'norm']].values)):
            with tf.name_scope('conv%d' % i):
                layer = {}
                image = tf.nn.conv2d(image, weight, strides=[1, stride1, stride2, 1], padding='SAME')
                layer['conv'] = image
                image = tf.nn.bias_add(image, bais)
                layer['add'] = image
                if norm == 'bn':
                    image = tf.layers.batch_normalization(image, training=training)
                    layer['norm'] = image
                elif norm == 'lrn':
                    image = tf.nn.lrn(image, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
                    layer['norm'] = image
                if act == 'relu':
                    image = tf.nn.relu(image)
                    layer['act'] = image
                elif act == 'lrelu':
                    image = tf.maximum(.1 * image, image, name='lrelu')
                    layer['act'] = image
                if pooling1 > 0 and pooling2 > 0:
                    image = tf.nn.max_pool(image, ksize=[1, pooling1, pooling2, 1], strides=[1, pooling1, pooling2, 1], padding='SAME')
                    layer['pool'] = image
                layer['output'] = image
                list.append(self, layer)
        self.output = image


class ModelFC(list):
    def __init__(self, data, param, layers, training=False, seed=None):
        for i, ((weight, bais), (act, norm, dropout)) in enumerate(zip(param, layers[['act', 'norm', 'dropout']].values)):
            with tf.name_scope('fc%d' % i):
                layer = {}
                data = tf.matmul(data, weight)
                layer['matmul'] = data
                data = data + bais
                layer['add'] = data
                if norm == 'bn':
                    data = tf.layers.batch_normalization(data, training=training)
                    layer['norm'] = data
                elif norm == 'lrn':
                    data = tf.nn.lrn(data, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
                    layer['norm'] = data
                if act == 'relu':
                    data = tf.nn.relu(data)
                    layer['act'] = data
                elif act == 'lrelu':
                    data = tf.maximum(.1 * data, data, name='lrelu')
                    layer['act'] = data
                if 0 < dropout < 1 and training:
                    data = tf.nn.dropout(data, dropout, seed=seed)
                    layer['dropout'] = data
                layer['output'] = data
                list.append(self, layer)
        self.output = data
        self.param = param
        self.training = training
        self.seed = seed
    
    def __call__(self, weight, bais):
        with tf.name_scope('fc'):
            layer = {}
            data = tf.matmul(self.output, weight)
            layer['matmul'] = data
            data = data + bais
            layer['add'] = data
            layer['output'] = data
            list.append(self, layer)
        self.output = data
