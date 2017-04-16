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
        for i, (size, kernel1, kernel2, norm) in enumerate(layers[['size', 'kernel1', 'kernel2', 'norm']].values):
            with tf.variable_scope('conv%d' % i):
                param = {}
                weight = tf.Variable(tf.truncated_normal([kernel1, kernel2, channels, size], stddev=1.0 / math.sqrt(channels * kernel1 * kernel2), seed=seed), name='weight')
                param['weight'] = weight
                bais = tf.Variable(tf.zeros([size]), name='bais')
                param['bais'] = bais
                if norm == 'bn':
                    scale = tf.Variable(tf.ones([size]), name='scale')
                    param['scale'] = scale
                list.append(self, param)
                channels = size
    
    def __call__(self, outputs, kernel1=1, kernel2=1, seed=None):
        channels = self.get_size(-1)
        with tf.variable_scope('conv'):
            param = {}
            weight = tf.Variable(tf.truncated_normal([kernel1, kernel2, channels, outputs], stddev=1.0 / math.sqrt(channels * kernel1 * kernel2), seed=seed), name='weight')
            param['weight'] = weight
            bais = tf.Variable(tf.zeros([outputs]), name='bais')
            param['bais'] = bais
            list.append(self, param)
    
    def get_size(self, i):
        return list.__getitem__(self, i)['bais'].get_shape()[0].value


class ParamFC(list):
    def __init__(self, inputs, layers, seed=None):
        for i, (size, norm) in enumerate(layers[['size', 'norm']].values):
            with tf.variable_scope('fc%d' % i):
                param = {}
                weight = tf.Variable(tf.truncated_normal([inputs, size], stddev=1.0 / math.sqrt(inputs), seed=seed), name='weight')
                param['weight'] = weight
                bais = tf.Variable(tf.zeros([size]), name='bais')
                param['bais'] = bais
                if norm == 'bn':
                    scale = tf.Variable(tf.ones([size]), name='scale')
                    param['scale'] = scale
                list.append(self, param)
                inputs = size
        self.seed = seed
    
    def __call__(self, outputs):
        inputs = self.get_size(-1)
        with tf.variable_scope('fc'):
            param = {}
            weight = tf.Variable(tf.truncated_normal([inputs, outputs], stddev=1.0 / math.sqrt(inputs), seed=self.seed), name='weight')
            param['weight'] = weight
            bais = tf.Variable(tf.zeros([outputs]), name='bais')
            param['bais'] = bais
            list.append(self, param)
    
    def get_size(self, i):
        return list.__getitem__(self, i)['bais'].get_shape()[0].value


class ModelConv(list):
    def __init__(self, image, param, layers, training=False, seed=None):
        for i, (param, (stride1, stride2, pooling1, pooling2, act, norm)) in enumerate(zip(param, layers[['stride1', 'stride2', 'pooling1', 'pooling2', 'act', 'norm']].values)):
            scope = 'conv%d' % i
            with tf.name_scope(scope):
                layer = {'image': image}
                image = tf.nn.conv2d(image, param['weight'], strides=[1, stride1, stride2, 1], padding='SAME')
                layer['conv'] = image
                if norm == 'bn':
                    epsilon = 1e-3
                    if training:
                        image = tf.nn.batch_normalization(image, *tf.nn.moments(image, [0]), param['bais'], param['scale'], epsilon)
                    else:
                        size = param['bais'].get_shape()[-1].value
                        image = tf.nn.batch_normalization(image, tf.zeros([size]), tf.ones([size]), param['bais'], param['scale'], epsilon)
                    layer['norm'] = image
                else:
                    image = tf.nn.bias_add(image, param['bais'])
                    layer['add'] = image
                if act == 'relu':
                    image = tf.nn.relu(image)
                    layer['act'] = image
                elif act == 'lrelu':
                    with tf.name_scope('lrelu'):
                        image = tf.maximum(.1 * image, image, name='lrelu')
                    layer['act'] = image
                if pooling1 > 0 and pooling2 > 0:
                    image = tf.nn.max_pool(image, ksize=[1, pooling1, pooling2, 1], strides=[1, pooling1, pooling2, 1], padding='SAME')
                    layer['pool'] = image
                layer['output'] = image
                list.append(self, layer)
        self.output = image
    
    def __call__(self, param, strides=[1, 1, 1, 1], padding='SAME'):
        with tf.name_scope('conv'):
            layer = {}
            image = tf.nn.conv2d(self.output, param['weight'], strides=strides, padding=padding)
            layer['conv'] = image
            image = tf.nn.bias_add(image, param['bais'])
            layer['add'] = image
            layer['output'] = image
            list.append(self, layer)
        self.output = image


class ModelFC(list):
    def __init__(self, data, param, layers, training=False, seed=None):
        for i, (param, (act, norm, dropout)) in enumerate(zip(param, layers[['act', 'norm', 'dropout']].values)):
            scope = 'fc%d' % i
            with tf.name_scope(scope):
                layer = {'data': data}
                data = tf.matmul(data, param['weight'])
                layer['matmul'] = data
                if norm == 'bn':
                    epsilon = 1e-3
                    if training:
                        data = tf.nn.batch_normalization(data, *tf.nn.moments(data, [0]), param['bais'], param['scale'], epsilon)
                    else:
                        size = param['bais'].get_shape()[-1].value
                        data = tf.nn.batch_normalization(data, tf.zeros([size]), tf.ones([size]), param['bais'], param['scale'], epsilon)
                    layer['norm'] = data
                else:
                    data = data + param['bais']
                    layer['add'] = data
                if act == 'relu':
                    data = tf.nn.relu(data)
                    layer['act'] = data
                elif act == 'lrelu':
                    with tf.name_scope('lrelu'):
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
    
    def __call__(self, param):
        with tf.name_scope('fc'):
            layer = {}
            data = tf.matmul(self.output, param['weight'])
            layer['matmul'] = data
            data = data + param['bais']
            layer['add'] = data
            layer['output'] = data
            list.append(self, layer)
        self.output = data
