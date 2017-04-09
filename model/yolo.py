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
import numpy as np
import tensorflow as tf
import model


def transform_labels_voc(imageshapes, labels, width, height, cell_width, cell_height, boxes_per_cell, classes):
    cells = cell_height * cell_width
    mask = np.zeros([len(labels), cells, 1])
    pred = np.zeros([len(labels), cells, classes])
    coords = np.zeros([len(labels), cells, boxes_per_cell, 4])
    offset_xy_min = np.zeros([len(labels), cells, boxes_per_cell, 2])
    offset_xy_max = np.zeros([len(labels), cells, boxes_per_cell, 2])
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
            pred[i, index, :] = [0] * classes
            pred[i, index, c] = 1
            coords[i, index, :, :] = [[offset_x, offset_y, math.sqrt(_w), math.sqrt(_h)]] * boxes_per_cell
            offset_xy_min[i, index, :, :] = [[offset_x - _w / 2 * cell_width, offset_y - _h / 2 * cell_height]] * boxes_per_cell
            offset_xy_max[i, index, :, :] = [[offset_x + _w / 2 * cell_width, offset_y + _h / 2 * cell_height]] * boxes_per_cell
    wh = offset_xy_max - offset_xy_min
    assert np.all(wh >= 0)
    areas = np.multiply.reduce(wh, -1)
    return mask, pred, coords, offset_xy_min, offset_xy_max, areas


class Model(object):
    def __init__(self, image, param_conv, param_fc, layers_conv, layers_fc, classes, boxes_per_cell, training=False, seed=None):
        self.image = image
        self.conv = model.ModelConv(self.image, param_conv, layers_conv, training, seed)
        data_fc = tf.reshape(self.conv.output, [self.conv.output.get_shape()[0].value, -1], name='data_fc')
        self.fc = model.ModelFC(data_fc, param_fc, layers_fc, training, seed)
        self.fc(*param_fc[-1])
        _, cell_height, cell_width, _ = self.conv.output.get_shape().as_list()
        cells = cell_height * cell_width
        with tf.name_scope('output'):
            end = cells * classes
            self.pred = tf.reshape(self.fc.output[:, :end], [-1, cells, classes], name='pred')
            start = end
            end += cells * boxes_per_cell
            self.iou = tf.reshape(self.fc.output[:, start:end], [-1, cells, boxes_per_cell], name='iou')
            with tf.name_scope('offset'):
                start = end
                end += cells * boxes_per_cell * 2
                self.offset_xy = tf.reshape(self.fc.output[:, start:end], [-1, cells, boxes_per_cell, 2], name='offset_xy')
                wh01_sqrt_base = tf.reshape(self.fc.output[:, end:], [-1, cells, boxes_per_cell, 2], name='wh01_sqrt_base')
                wh01 = wh01_sqrt_base ** 2
                wh01_sqrt = tf.abs(wh01_sqrt_base, name='wh01_sqrt')
                self.coords = tf.concat([self.offset_xy, wh01_sqrt], -1, name='coords')
                self.wh = wh01 * [cell_width, cell_height]
                _wh = self.wh / 2
                self.offset_xy_min = self.offset_xy - _wh
                self.offset_xy_max = self.offset_xy + _wh
                self.areas = self.wh[:, :, :, 0] * self.wh[:, :, :, 1]
            with tf.name_scope('xy'):
                cell_xy = self.calc_cell_xy().reshape([1, cells, 1, 2])
                self.xy = cell_xy + self.offset_xy
                self.xy_min = cell_xy + self.offset_xy_min
                self.xy_max = cell_xy + self.offset_xy_max
            self.prob = tf.reshape(self.pred, [-1, cells, 1, classes]) * tf.expand_dims(self.iou, -1)
        self.regularizer = tf.reduce_sum([tf.nn.l2_loss(weight) for weight, _ in param_fc], name='regularizer')
        self.param_conv = param_conv
        self.param_fc = param_fc
        self.classes = classes
        self.boxes_per_cell = boxes_per_cell
    
    def calc_cell_xy(self):
        _, cell_height, cell_width, _ = self.conv.output.get_shape().as_list()
        cell_base = np.zeros([cell_height, cell_width, 2])
        for y in range(cell_height):
            for x in range(cell_width):
                cell_base[y, x, :] = [x, y]
        return cell_base


class Loss(dict):
    def __init__(self, model, mask, pred, coords, offset_xy_min, offset_xy_max, areas):
        self.model = model
        self.mask = mask
        self.pred = pred
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
            mask_best = self.mask * mask_max_iou
            mask_normal = 1 - mask_best
        iou_diff = model.iou - iou
        with tf.name_scope('objectives'):
            self['pred'] = tf.nn.l2_loss(self.mask * model.pred - self.pred, name='pred')
            self['iou_best'] = tf.nn.l2_loss(mask_best * iou_diff, name='mask_best')
            self['iou_normal'] = tf.nn.l2_loss(mask_normal * iou_diff, name='mask_normal')
            self['coords'] = tf.nn.l2_loss(tf.expand_dims(mask_best, -1) * (model.coords - self.coords), name='coords')
