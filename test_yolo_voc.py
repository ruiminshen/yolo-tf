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
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from tensorflow.python.framework import ops
import yolo as model
import utils


def check_mark(index, names, cell_width, cell_height, image_width, image_height, ax, mask, pred, coords, xy_min, xy_max, areas):
    coords = coords[0]
    xy_min = xy_min[0]
    xy_max = xy_max[0]
    areas = areas[0]
    if np.any(mask) > 0:
        iy = index // cell_width
        ix = index % cell_width
        name = names[np.argmax(pred)]
        #check coords
        offset_x, offset_y, _w_sqrt, _h_sqrt = coords
        cell_x, cell_y = ix + offset_x, iy + offset_y
        x, y = cell_x * image_width / cell_width, cell_y * image_height / cell_height
        _w, _h = _w_sqrt ** 2, _h_sqrt ** 2
        w, h = _w * image_width, _h * image_height
        x_min, y_min = x - w / 2, y - h / 2
        ax.add_patch(patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor='r', facecolor='none'))
        ax.annotate(name, (x_min, y_min), color='red')
        #check xy_min and xy_max
        wh = xy_max - xy_min
        np.testing.assert_allclose(wh / [cell_width, cell_height], [_w, _h])
        np.testing.assert_allclose(xy_min + wh / 2, [offset_x, offset_y])


def _show(names, cell_width, cell_height, image, ax, mask, pred, coords, xy_min, xy_max, areas):
    image_height, image_width, _ = image.shape
    vmin = np.min(image)
    vmax = np.max(image)
    image = ((image - vmin) * 255 / (vmax - vmin)).astype(np.uint8)
    ax.imshow(image)
    for index, results in enumerate(zip(mask, pred, coords, xy_min, xy_max, areas)):
        check_mark(index, names, cell_width, cell_height, image_width, image_height, ax, *results)
    ax.set_xticks([])
    ax.set_yticks([])


def show(names, cell_width, cell_height, image, mask, pred, coords, xy_min, xy_max, areas):
    row, col = utils.get_factor2(len(image))
    fig, axes = plt.subplots(row, col)
    for results in zip(image, axes.flat, mask, pred, coords, xy_min, xy_max, areas):
        _show(names, cell_width, cell_height, *results)
    return fig


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    with open(os.path.expanduser(os.path.expandvars(config.get('yolo', 'names'))), 'r') as f:
        names = [line.strip() for line in f]
    path = os.path.expanduser(os.path.expandvars(config.get(model.__name__, 'cache')))
    with open(path, 'rb') as f:
        data = pickle.load(f)
    width = config.getint('yolo', 'width')
    height = config.getint('yolo', 'height')
    layers_conv = pd.read_csv(os.path.expanduser(os.path.expandvars(config.get('yolo', 'conv'))), sep='\t')
    cell_width = utils.calc_pooled_size(width, layers_conv['pooling1'].values)
    cell_height = utils.calc_pooled_size(height, layers_conv['pooling2'].values)
    with tf.Session() as sess:
        with tf.name_scope('data'):
            with tf.device('/cpu:0'):
                imagepaths = tf.train.string_input_producer(data[0], shuffle=False)
                reader = tf.WholeFileReader()
                _, image = reader.read(imagepaths)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.image.resize_images(image, [height, width])
                image = tf.image.per_image_standardization(image)
                labels = [ops.convert_to_tensor(l) for l in data[1:]]
                labels = tf.train.slice_input_producer(labels, shuffle=False)
                data = tf.train.shuffle_batch([image] + labels, batch_size=args.batch_size, capacity=args.batch_size * config.getint('batch', 'capacity'), min_after_dequeue=args.batch_size * config.getint('batch', 'min'), num_threads=multiprocessing.cpu_count())
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        _data = sess.run(data)
        coord.request_stop()
        coord.join(threads)
    fig = show(names, cell_width, cell_height, *_data)
    fig.tight_layout()
    plt.show()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.ini', help='config file')
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    assert os.path.exists(args.config)
    config.read(args.config)
    main()
