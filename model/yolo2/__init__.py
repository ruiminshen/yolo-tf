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

import configparser
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import utils
from . import inference
from .. import yolo


class Model(object):
    def __init__(self, net, classes, anchors):
        _, self.cell_height, self.cell_width, _ = net.get_shape().as_list()
        cells = self.cell_height * self.cell_width
        inputs = tf.reshape(net, [-1, cells, len(anchors), 5 + classes], name='inputs')
        with tf.name_scope('regress'):
            with tf.name_scope('inputs'):
                _inputs = tf.nn.sigmoid(inputs[:, :, :, :3])
                self.iou = _inputs[:, :, :, 0]
                self.offset_xy = tf.identity(_inputs[:, :, :, 1:3], name='offset_xy')
                self.wh = tf.identity(tf.exp(inputs[:, :, :, 3:5]) * np.reshape(anchors, [1, 1, len(anchors), -1]), name='wh')
                prob = inputs[:, :, :, 5:]
                self.prob = tf.reshape(tf.nn.softmax(prob), prob.get_shape(), name='prob')
            self.areas = tf.identity(self.wh[:, :, :, 0] * self.wh[:, :, :, 1], name='areas')
            _wh = self.wh / 2
            self.offset_xy_min = tf.identity(self.offset_xy - _wh, name='offset_xy_min')
            self.offset_xy_max = tf.identity(self.offset_xy + _wh, name='offset_xy_max')
            self.wh01 = tf.identity(self.wh / np.reshape([self.cell_width, self.cell_height], [1, 1, 1, 2]), name='wh01')
            self.wh01_sqrt = tf.sqrt(self.wh01, name='wh01_sqrt')
            self.coords = tf.concat([self.offset_xy, self.wh01_sqrt], -1, name='coords')
        with tf.name_scope('detection'):
            cell_xy = yolo.calc_cell_xy(self.cell_height, self.cell_width).reshape([1, cells, 1, 2])
            self.xy = tf.identity(cell_xy + self.offset_xy, name='xy')
            self.xy_min = tf.identity(cell_xy + self.offset_xy_min, name='xy_min')
            self.xy_max = tf.identity(cell_xy + self.offset_xy_max, name='xy_max')
            self.conf = tf.identity(tf.expand_dims(self.iou, -1) * self.prob, name='conf')
        self.inputs = net
        self.classes = classes
        self.anchors = anchors


class Objectives(dict):
    def __init__(self, model, mask, prob, coords, offset_xy_min, offset_xy_max, areas):
        self.model = model
        with tf.name_scope('true'):
            self.mask = tf.identity(mask, name='mask')
            self.prob = tf.identity(prob, name='prob')
            self.coords = tf.identity(coords, name='coords')
            self.offset_xy_min = tf.identity(offset_xy_min, name='offset_xy_min')
            self.offset_xy_max = tf.identity(offset_xy_max, name='offset_xy_max')
            self.areas = tf.identity(areas, name='areas')
        with tf.name_scope('iou') as name:
            _offset_xy_min = tf.maximum(model.offset_xy_min, self.offset_xy_min) 
            _offset_xy_max = tf.minimum(model.offset_xy_max, self.offset_xy_max)
            _wh = tf.maximum(_offset_xy_max - _offset_xy_min, 0.0)
            _areas = _wh[:, :, :, 0] * _wh[:, :, :, 1]
            areas = tf.maximum(self.areas + model.areas - _areas, 1e-10)
            iou = tf.truediv(_areas, areas, name=name)
        with tf.name_scope('mask'):
            best_box_iou = tf.reduce_max(iou, 2, True, name='best_box_iou')
            best_box = tf.to_float(tf.equal(iou, best_box_iou, name='best_box'))
            mask_best = tf.identity(self.mask * best_box, name='mask_best')
            mask_normal = tf.identity(1 - mask_best, name='mask_normal')
        with tf.name_scope('diff2'):
            iou_diff2 = tf.pow(model.iou - mask_best, 2, name='iou_diff2')
            coords_diff2 = tf.pow(model.coords - self.coords, 2, name='coords_diff2')
            prob_diff2 = tf.pow(model.prob - self.prob, 2, name='prob_diff2')
        with tf.name_scope('objectives'):
            self['iou_best'] = tf.reduce_sum(mask_best * iou_diff2, name='iou_best')
            self['iou_normal'] = tf.reduce_sum(mask_normal * iou_diff2, name='iou_normal')
            _mask_best = tf.expand_dims(mask_best, -1)
            self['coords'] = tf.reduce_sum(_mask_best * coords_diff2, name='coords')
            self['prob'] = tf.reduce_sum(_mask_best * prob_diff2, name='prob')


class Builder(yolo.Builder):
    def __init__(self, args, config):
        section = __name__.split('.')[-1]
        self.args = args
        self.config = config
        with open(os.path.join(utils.get_cachedir(config), 'names'), 'r') as f:
            self.names = [line.strip() for line in f]
        self.width = config.getint(section, 'width')
        self.height = config.getint(section, 'height')
        self.anchors = pd.read_csv(os.path.expanduser(os.path.expandvars(config.get(section, 'anchors'))), sep='\t').values
        self.func = getattr(inference, config.get(section, 'inference'))
    
    def __call__(self, data, training=False):
        _, self.output = self.func(data, len(self.names), len(self.anchors), training=training)
        with tf.name_scope(__name__.split('.')[-1]):
            self.model = Model(self.output, len(self.names), self.anchors)
    
    def loss(self, labels):
        section = __name__.split('.')[-1]
        with tf.name_scope('loss') as name:
            self.objectives = Objectives(self.model, *labels)
            with tf.variable_scope('hparam'):
                self.hparam = dict([(key, tf.Variable(float(s), name='hparam_' + key, trainable=False)) for key, s in self.config.items(section + '_hparam')])
            with tf.name_scope('loss_objectives'):
                loss_objectives = tf.reduce_sum([tf.multiply(self.objectives[key], self.hparam[key], name='weighted_' + key) for key in self.objectives], name='loss_objectives')
            loss = tf.identity(loss_objectives, name=name)
        return loss
