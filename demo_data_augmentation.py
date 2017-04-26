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
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import utils


def main():
    section = config.get('config', 'model')
    path = os.path.expanduser(os.path.expandvars(config.get(section, 'cache')))
    logger.info('loading cache from ' + path)
    with open(path, 'rb') as f:
        data_labels = pickle.load(f)
    logger.info('size: %d (batch size: %d)' % (len(data_labels[0]), args.batch_size))
    width = config.getint(section, 'width')
    height = config.getint(section, 'height')
    with tf.Session() as sess:
        with tf.name_scope('data_labels'):
            with tf.device('/cpu:0'):
                paths = tf.train.string_input_producer(data_labels[0], shuffle=False)
                reader = tf.WholeFileReader()
                _, data = reader.read(paths)
                data = tf.image.decode_jpeg(data, channels=3)
                data = tf.image.resize_images(data, [height, width])
                labels = [ops.convert_to_tensor(l, dtype=tf.float32) for l in data_labels[1:]]
                labels = tf.train.slice_input_producer(labels, shuffle=False)
                if config.getboolean('data_augmentation', 'enable'):
                    data, labels = utils.data_augmentation(data, labels, config)
                #data = tf.image.per_image_standardization(data)
                data_labels = tf.train.shuffle_batch([data] + labels, batch_size=args.batch_size, capacity=config.getint('queue', 'capacity'), min_after_dequeue=config.getint('queue', 'min_after_dequeue'), num_threads=multiprocessing.cpu_count())
            data = tf.identity(data_labels[0], name='data')
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        images, labels = sess.run([data, data_labels[1:]])
        coord.request_stop()
        coord.join(threads)
    """
    vmin = np.min(images, (1, 2, 3)).reshape([args.batch_size, 1, 1, 1])
    vmax = np.max(images, (1, 2, 3)).reshape([args.batch_size, 1, 1, 1])
    images = ((images - vmin) * 255 / (vmax - vmin)).astype(np.uint8)
    """
    print(np.min(images), np.max(images))
    images = images.astype(np.uint8)
    row, col = utils.get_factor2(args.batch_size)
    fig, axes = plt.subplots(row, col)
    for ax, _image in zip(axes.flat, images):
        ax.imshow(_image)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.ini', help='config file')
    parser.add_argument('-l', '--level', default='info', help='logging level')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='batch size')
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
