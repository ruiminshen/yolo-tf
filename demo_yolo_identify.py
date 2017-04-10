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
import importlib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from tensorflow.python.framework import ops
import model
import utils


class Drawer(object):
    def __init__(self, sess, names, cell_width, cell_height, image, labels, model, loss):
        self.sess = sess
        self.names = names
        self.cell_width, self.cell_height = cell_width, cell_height
        self.image, self.labels = image, labels
        self.model, self.loss = model, loss
        self.fig = plt.figure()
        self.ax = self.fig.gca()
        image_height, image_width, _ = image.shape
        vmin = np.min(image)
        vmax = np.max(image)
        image = ((image - vmin) * 255 / (vmax - vmin)).astype(np.uint8)
        self.ax.imshow(image)
        self.plots = []
        for index, label in enumerate(zip(*labels)):
            self.plots += draw_label(self.ax, names, cell_width, cell_height, image_width, image_height, index, *label)
        self.ax.set_xticks(np.arange(0, image_width, image_width / cell_width))
        self.ax.set_yticks(np.arange(0, image_height, image_height / cell_height))
        self.ax.grid(which='both')
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.colors = plt.rcParams['axes.color_cycle']
    
    def onclick(self, event):
        for p in self.plots:
            p.remove()
        self.plots = []
        image_height, image_width, _ = self.image.shape
        ix = int(event.xdata * self.cell_width / image_width)
        iy = int(event.ydata * self.cell_height / image_height)
        index = iy * self.cell_width + ix
        pred, iou, xy_min, wh = self.sess.run([self.model.pred[0][index], self.model.iou[0][index], self.model.xy_min[0][index], self.model.wh[0][index]])
        xy_min = xy_min * [image_width, image_height] / [self.cell_width, self.cell_height]
        wh = wh * [image_width, image_height] / [self.cell_width, self.cell_height]
        name = self.names[np.argmax(pred)]
        self.fig.suptitle(name)
        for color, conf, (x, y), (w, h) in zip(self.colors, iou, xy_min, wh):
            self.plots.append(self.ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')))
            self.plots.append(self.ax.annotate(str(conf), (x, y), color=color))
        self.fig.canvas.draw()


def draw_label(ax, names, cell_width, cell_height, image_width, image_height, index, mask, pred, coords, xy_min, xy_max, areas, rtol=1e-5):
    coords = coords[0]
    xy_min = xy_min[0]
    xy_max = xy_max[0]
    areas = areas[0]
    plots = []
    if np.any(mask) > 0:
        iy = index // cell_width
        ix = index % cell_width
        plots.append(ax.add_patch(patches.Rectangle((ix * image_width / cell_width, iy * image_height / cell_height), image_width / cell_width, image_height / cell_height, linewidth=0, facecolor='red', alpha=.2)))
        name = names[np.argmax(pred)]
        #check coords
        offset_x, offset_y, _w_sqrt, _h_sqrt = coords
        cell_x, cell_y = ix + offset_x, iy + offset_y
        x, y = cell_x * image_width / cell_width, cell_y * image_height / cell_height
        _w, _h = _w_sqrt ** 2, _h_sqrt ** 2
        w, h = _w * image_width, _h * image_height
        x_min, y_min = x - w / 2, y - h / 2
        plots.append(ax.add_patch(patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor='r', facecolor='none')))
        plots.append(ax.annotate(name, (x_min, y_min), color='red'))
        #check offset_xy_min and xy_max
        wh = xy_max - xy_min
        np.testing.assert_allclose(wh / [cell_width, cell_height], [_w, _h], rtol=rtol)
        np.testing.assert_allclose(xy_min + wh / 2, [offset_x, offset_y], rtol=rtol)
    return plots


def main():
    section = config.get('config', 'model')
    yolo = importlib.import_module('model.' + section)
    yolodir = os.path.expanduser(os.path.expandvars(config.get('yolo', 'dir')))
    modeldir = os.path.join(yolodir, 'model')
    path_model = os.path.join(modeldir, 'model.ckpt')
    with open(os.path.expanduser(os.path.expandvars(config.get(section, 'names'))), 'r') as f:
        names = [line.strip() for line in f]
    path = os.path.expanduser(os.path.expandvars(config.get(section, 'cache')))
    logger.info('loading cache from ' + path)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    logger.info('size: %d' % len(data[0]))
    width = config.getint(section, 'width')
    height = config.getint(section, 'height')
    with tf.Session() as sess:
        with tf.name_scope('data'):
            with tf.device('/cpu:0'):
                imagepath = data[0][args.index]
                logger.info('imagepath=' + imagepath)
                imagepath = ops.convert_to_tensor(imagepath)
                image = tf.image.decode_jpeg(tf.read_file(imagepath), channels=3)
                image = tf.image.resize_images(image, [height, width])
                image = tf.image.per_image_standardization(image)
                image = tf.expand_dims(image, 0)
                labels = [tf.expand_dims(ops.convert_to_tensor(l[args.index], dtype=tf.float32), 0) for l in data[1:]]
        modeler = yolo.Modeler(args, config)
        modeler.param()
        modeler.eval(image)
        with tf.name_scope('loss'):
            loss = yolo.Loss(modeler.model_eval, *labels)
        tf.global_variables_initializer().run()
        logger.info('load model')
        saver = tf.train.Saver()
        saver.restore(sess, path_model)
        _ = Drawer(sess, names, modeler.cell_width, modeler.cell_height, sess.run(image[0]), sess.run([l[0] for l in labels]), modeler.model_eval, loss)
        plt.show()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', type=int, default=0)
    parser.add_argument('-c', '--config', default='config.ini', help='config file')
    parser.add_argument('-l', '--level', default='info', help='logging level')
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
