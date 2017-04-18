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

import os
import math
import configparser
import numpy as np
import pandas as pd
import tensorflow as tf
import model
import utils


def transform_labels_voc(imageshapes, labels, width, height, cell_width, cell_height, classes):
    cells = cell_height * cell_width
    mask = np.zeros([len(labels), cells, 1])
    prob = np.zeros([len(labels), cells, 1, classes])
    coords = np.zeros([len(labels), cells, 1, 4])
    offset_xy_min = np.zeros([len(labels), cells, 1, 2])
    offset_xy_max = np.zeros([len(labels), cells, 1, 2])
    for i, ((image_height, image_width, _), objects) in enumerate(zip(imageshapes, labels)):
        for xmin, ymin, xmax, ymax, c in objects:
            x = (xmin + xmax) / 2
            y = (ymin + ymax) / 2
            cell_x = x * cell_width / image_width
            cell_y = y * cell_height / image_height
            assert 0 <= cell_x < cell_width
            assert 0 <= cell_y < cell_height
            ix = math.floor(cell_x)
            iy = math.floor(cell_y)
            index = iy * cell_width + ix
            offset_x = cell_x - ix
            offset_y = cell_y - iy
            _w = float(xmax - xmin) / image_width
            _h = float(ymax - ymin) / image_height
            mask[i, index, :] = 1
            prob[i, index, 0, :] = [0] * classes
            prob[i, index, 0, c] = 1
            coords[i, index, 0, :] = [offset_x, offset_y, math.sqrt(_w), math.sqrt(_h)]
            offset_xy_min[i, index, 0, :] = [offset_x - _w / 2 * cell_width, offset_y - _h / 2 * cell_height]
            offset_xy_max[i, index, 0, :] = [offset_x + _w / 2 * cell_width, offset_y + _h / 2 * cell_height]
    wh = offset_xy_max - offset_xy_min
    assert np.all(wh >= 0)
    areas = np.multiply.reduce(wh, -1)
    return mask, prob, coords, offset_xy_min, offset_xy_max, areas


class Model(object):
    def __init__(self, image, param, layers_conv, classes, anchors, training=False, seed=None):
        self.image = image
        self.conv = model.ModelConv(self.image, param, layers_conv, training, seed)
        self.conv(param[-1])
        boxes_per_cell, _ = anchors.shape
        _, cell_height, cell_width, _ = self.conv.output.get_shape().as_list()
        cells = cell_height * cell_width
        output = tf.reshape(self.conv.output, [-1, cells, boxes_per_cell, 5 + classes], name='output')
        with tf.name_scope('labels'):
            output_sigmoid = tf.nn.sigmoid(output[:, :, :, :3])
            end = 1
            self.iou = output_sigmoid[:, :, :, end]
            start = end
            end += 2
            self.offset_xy = tf.identity(output_sigmoid[:, :, :, start:end], name='offset_xy')
            start = end
            end += 2
            self.wh = tf.identity(tf.exp(output[:, :, :, start:end]) * np.reshape(anchors, [1, 1, boxes_per_cell, -1]), name='wh')
            self.areas = tf.identity(self.wh[:, :, :, 0] * self.wh[:, :, :, 1], name='areas')
            _wh = self.wh / 2
            self.offset_xy_min = tf.identity(self.offset_xy - _wh, name='offset_xy_min')
            self.offset_xy_max = tf.identity(self.offset_xy + _wh, name='offset_xy_max')
            self.wh01 = tf.identity(self.wh / np.reshape([cell_width, cell_height], [1, 1, 1, 2]), name='wh01')
            self.wh01_sqrt = tf.sqrt(self.wh01, name='wh01_sqrt')
            self.coords = tf.concat([self.offset_xy, self.wh01_sqrt], -1, name='coords')
            self.prob = tf.nn.softmax(output[:, :, :, end:])
        with tf.name_scope('xy'):
            cell_xy = self.calc_cell_xy(cell_height, cell_width).reshape([1, cells, 1, 2])
            self.xy = tf.identity(cell_xy + self.offset_xy, name='xy')
            self.xy_min = tf.identity(cell_xy + self.offset_xy_min, name='xy_min')
            self.xy_max = tf.identity(cell_xy + self.offset_xy_max, name='xy_max')
        self.conf = tf.identity(self.prob * tf.expand_dims(self.iou, -1), name='conf')
        self.regularizer = tf.reduce_sum([tf.nn.l2_loss(p['weight']) for p in param], name='regularizer')
        self.param = param
        self.classes = classes
        self.anchors = anchors
    
    def calc_cell_xy(self, cell_height, cell_width):
        cell_base = np.zeros([cell_height, cell_width, 2])
        for y in range(cell_height):
            for x in range(cell_width):
                cell_base[y, x, :] = [x, y]
        return cell_base


class Loss(dict):
    def __init__(self, model, mask, prob, coords, offset_xy_min, offset_xy_max, areas):
        self.model = model
        self.mask = mask
        self.prob = prob
        self.coords = coords
        self.offset_xy_min = offset_xy_min
        self.offset_xy_max = offset_xy_max
        self.areas = areas
        with tf.name_scope('iou'):
            _offset_xy_min = tf.maximum(model.offset_xy_min, self.offset_xy_min) 
            _offset_xy_max = tf.minimum(model.offset_xy_max, self.offset_xy_max)
            _wh = tf.maximum(_offset_xy_max - _offset_xy_min, 0.0)
            _areas = _wh[:, :, :, 0] * _wh[:, :, :, 1]
            areas = tf.maximum(self.areas + model.areas - _areas, 1e-10)
            iou = tf.truediv(_areas, areas, name='iou')
        with tf.name_scope('mask'):
            max_iou = tf.reduce_max(iou, 2, True, name='max_iou')
            mask_max_iou = tf.to_float(tf.equal(iou, max_iou, name='mask_max_iou'))
            mask_best = tf.identity(self.mask * mask_max_iou, name='mask_best')
            mask_normal = tf.identity(1 - mask_best, name='mask_normal')
        iou_diff = model.iou - iou
        with tf.name_scope('objectives'):
            self['prob'] = tf.nn.l2_loss(tf.expand_dims(self.mask, -1) * model.prob - self.prob, name='prob')
            self['iou_best'] = tf.nn.l2_loss(mask_best * iou_diff, name='mask_best')
            self['iou_normal'] = tf.nn.l2_loss(mask_normal * iou_diff, name='mask_normal')
            self['coords'] = tf.nn.l2_loss(tf.expand_dims(mask_best, -1) * (model.coords - self.coords), name='coords')


class Modeler(object):
    def __init__(self, args, config):
        section = __name__.split('.')[-1]
        self.args = args
        self.config = config
        with open(os.path.expanduser(os.path.expandvars(config.get(section, 'names'))), 'r') as f:
            self.names = [line.strip() for line in f]
        self.width = config.getint(section, 'width')
        self.height = config.getint(section, 'height')
        self.layers_conv = pd.read_csv(os.path.expanduser(os.path.expandvars(config.get(section, 'conv'))), sep='\t')
        self.cell_width = utils.calc_pooled_size(self.width, self.layers_conv['pooling1'].values)
        self.cell_height = utils.calc_pooled_size(self.height, self.layers_conv['pooling2'].values)
        self.anchors = pd.read_csv(os.path.expanduser(os.path.expandvars(config.get(section, 'anchors'))), sep='\t').values
    
    def param(self, scope='param'):
        with tf.variable_scope(scope):
            self.param = model.ParamConv(3, self.layers_conv, seed=self.args.seed)
            boxes_per_cell = self.anchors.shape[0]
            outputs = boxes_per_cell * (5 + len(self.names))
            self.param(outputs)
    
    def train(self, image, labels, scope='train'):
        section = __name__.split('.')[-1]
        with tf.name_scope(scope):
            self.model_train = Model(image, self.param, self.layers_conv, len(self.names), self.anchors, training=True, seed=self.args.seed)
            with tf.name_scope('loss'):
                self.loss_train = Loss(self.model_train, *labels)
                with tf.variable_scope('hparam'):
                    self.hparam = dict([(key, tf.Variable(float(s), name='hparam_' + key, trainable=False)) for key, s in self.config.items(section + '_hparam')])
                    self.hparam_regularizer = tf.Variable(self.config.getfloat(section, 'hparam'), name='hparam_regularizer', trainable=False)
                self.loss = tf.reduce_sum([self.loss_train[key] * self.hparam[key] for key in self.loss_train], name='loss_objectives') + tf.multiply(self.hparam_regularizer, self.model_train.regularizer, name='loss_regularizer')
                for key in self.loss_train:
                    tf.summary.scalar(key, self.loss_train[key])
                tf.summary.scalar('regularizer', self.model_train.regularizer)
                tf.summary.scalar('loss', self.loss)
    
    def setup_histogram(self):
        if self.config.getboolean('histogram', 'param'):
            for param in self.param:
                for key, value in param.items():
                    try:
                        if self.config.getboolean('histogram_param_conv', key):
                            tf.summary.histogram(value.name, value)
                    except configparser.NoOptionError:
                        pass
        if self.config.getboolean('histogram', 'model'):
            for layer in self.model_train.conv:
                for key, value in layer.items():
                    try:
                        if self.config.getboolean('histogram_model_conv', key):
                            tf.summary.histogram(value.name, value)
                    except configparser.NoOptionError:
                        pass
    
    def log_hparam(self, sess, logger):
        keys, values = zip(*self.hparam.items())
        logger.info(', '.join(['%s=%f' % (key, value) for key, value in zip(keys, sess.run(values))]))
        logger.info('hparam_regularizer=%f' % sess.run(self.hparam_regularizer))
    
    def eval(self, image, scope='eval'):
        with tf.name_scope(scope):
            self.model_eval = Model(image, self.param, self.layers_conv, len(self.names), self.anchors)
