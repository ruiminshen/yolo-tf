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
import argparse
import configparser
import pickle
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from tensorflow.python.framework import ops
import model
import utils


def iou(xy_min1, xy_max1, xy_min2, xy_max2):
    assert np.all(xy_min1 <= xy_max1)
    assert np.all(xy_min2 <= xy_max2)
    areas1 = np.multiply.reduce(xy_max1 - xy_min1)
    areas2 = np.multiply.reduce(xy_max2 - xy_min2)
    _xy_min = np.maximum(xy_min1, xy_min2) 
    _xy_max = np.minimum(xy_max1, xy_max2)
    _wh = np.maximum(_xy_max - _xy_min, 0)
    _areas = np.multiply.reduce(_wh)
    assert _areas <= areas1
    assert _areas <= areas2
    return _areas / np.maximum(areas1 + areas2 - _areas, 1e-10)


def non_max_suppress(prob, xy_min, xy_max, threshold=.4):
    _, _, classes = prob.shape
    boxes = [(_prob, _xy_min, _xy_max) for _prob, _xy_min, _xy_max in zip(prob.reshape(-1, classes), xy_min.reshape(-1, 2), xy_max.reshape(-1, 2))]
    for c in range(classes):
        boxes.sort(key=lambda box: box[0][c], reverse=True)
        for i in range(len(boxes) - 1):
            box = boxes[i]
            if box[0][c] == 0:
                continue
            for _box in boxes[i + 1:]:
                if iou(box[1], box[2], _box[1], _box[2]) >= threshold:
                    _box[0][c] = 0
    return boxes


def main():
    section = config.get('config', 'model')
    yolo = importlib.import_module('model.' + section)
    yolodir = os.path.expanduser(os.path.expandvars(config.get('yolo', 'dir')))
    modeldir = os.path.join(yolodir, 'model')
    path_model = os.path.join(modeldir, 'model.ckpt')
    path = os.path.expanduser(os.path.expandvars(config.get(section, 'cache')))
    logger.info('loading cache from ' + path)
    with open(path, 'rb') as f:
        names = pickle.load(f)
    width = config.getint(section, 'width')
    height = config.getint(section, 'height')
    layers_conv = pd.read_csv(os.path.expanduser(os.path.expandvars(config.get(section, 'conv'))), sep='\t')
    cell_width = utils.calc_pooled_size(width, layers_conv['pooling1'].values)
    cell_height = utils.calc_pooled_size(height, layers_conv['pooling2'].values)
    layers_fc = pd.read_csv(os.path.expanduser(os.path.expandvars(config.get(section, 'fc'))), sep='\t')
    boxes_per_cell = config.getint(section, 'boxes_per_cell')
    with tf.Session() as sess:
        logger.info('init param')
        with tf.variable_scope('param'):
            param_conv = model.ParamConv(3, layers_conv, seed=args.seed)
            inputs = cell_width * cell_height * param_conv.bais[-1].get_shape()[0].value
            outputs = cell_width * cell_height * (len(names) + boxes_per_cell * 5)
            param_fc = model.ParamFC(inputs, layers_fc, outputs, seed=args.seed)
        with tf.name_scope('data'):
            image = tf.image.decode_jpeg(tf.read_file(ops.convert_to_tensor(args.image)), channels=3)
            image = tf.image.resize_images(image, [height, width])
            image = tf.image.per_image_standardization(image)
            image = tf.expand_dims(image, 0)
        logger.info('init model')
        with tf.name_scope('1'):
            model1 = yolo.Model(image, param_conv, param_fc, layers_conv, layers_fc, len(names), boxes_per_cell)
        tf.global_variables_initializer().run()
        logger.info('load model')
        saver = tf.train.Saver()
        saver.restore(sess, path_model)
        image = sess.run(image[0])
        vmin = np.min(image)
        vmax = np.max(image)
        image = ((image - vmin) * 255 / (vmax - vmin)).astype(np.uint8)
        fig = plt.figure()
        ax = fig.gca()
        ax.imshow(image)
        prob, xy_min, xy_max = sess.run([model1.prob * tf.to_float(model1.prob > args.threshold), model1.xy_min, model1.xy_max])
        boxes = non_max_suppress(prob[0], xy_min[0], xy_max[0])
        for _prob, _xy_min, _xy_max in boxes:
            index = np.argmax(_prob)
            if _prob[index] > args.threshold:
                wh = _xy_max - _xy_min
                _xy_min = _xy_min * [width, height] / [cell_width, cell_height]
                _wh = wh * [width, height] / [cell_width, cell_height]
                ax.add_patch(patches.Rectangle(_xy_min, _wh[0], _wh[1], linewidth=1, edgecolor='r', facecolor='none'))
                ax.annotate(names[index], _xy_min, color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='input image')
    parser.add_argument('-c', '--config', default='config.ini', help='config file')
    parser.add_argument('-l', '--level', default='info', help='logging level')
    parser.add_argument('-t', '--threshold', type=int, default=0.1, help='detection threshold')
    parser.add_argument('--seed', type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    assert os.path.exists(args.config)
    config.read(args.config)
    logger = utils.make_logger(importlib.import_module('logging').__dict__[args.level.strip().upper()], config.get('logging', 'format'))
    try:
        main()
    except Exception as e:
        logger.exception('exception')
        raise e
