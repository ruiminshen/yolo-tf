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
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import utils


def main():
    section = config.get('config', 'model')
    basedir = os.path.expanduser(os.path.expandvars(config.get(section, 'basedir')))
    with open(os.path.expanduser(os.path.expandvars(config.get(section, 'names'))), 'r') as f:
        names = [line.strip() for line in f]
    width = config.getint(section, 'width')
    height = config.getint(section, 'height')
    downsampling = config.getint(section, 'downsampling')
    assert width % downsampling == 0
    assert height % downsampling == 0
    cell_width, cell_height = width // downsampling, height // downsampling
    logger.info('(width, height)=(%d, %d), (cell_width, cell_height)=(%d, %d)' % (width, height, cell_width, cell_height))
    cachedir = os.path.join(basedir, 'cache')
    with tf.Session() as sess:
        with tf.name_scope('batch'):
            reader = tf.TFRecordReader()
            _, serialized = reader.read(tf.train.string_input_producer([os.path.join(cachedir, t + '.tfrecord') for t in args.types], shuffle=False))
            example = tf.parse_single_example(serialized, features={
                'imagepath': tf.FixedLenFeature([], tf.string),
                'objects': tf.FixedLenFeature([2], tf.string),
            })
            image_rgb, objects_class, objects_coord = utils.decode_image_objects(example, width, height)
            if config.getboolean('data_augmentation', 'enable'):
                image_rgb, objects_coord = utils.data_augmentation(image_rgb, objects_coord, config)
            #image_std = tf.image.per_image_standardization(image_rgb)
            labels = utils.decode_labels(objects_class, objects_coord, len(names), cell_width, cell_height)
            batch = tf.train.shuffle_batch((tf.cast(image_rgb, tf.uint8),) + labels, batch_size=args.batch_size, capacity=config.getint('queue', 'capacity'), min_after_dequeue=config.getint('queue', 'min_after_dequeue'), num_threads=multiprocessing.cpu_count())
            image = tf.identity(batch[0], name='image')
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        batch_image, batch_labels = sess.run([image, batch[1:]])
        coord.request_stop()
        coord.join(threads)
    print(np.min(batch_image), np.max(batch_image))
    batch_image = batch_image.astype(np.uint8)
    row, col = utils.get_factor2(args.batch_size)
    fig, axes = plt.subplots(row, col)
    for b, (ax, image) in enumerate(zip(axes.flat, batch_image)):
        ax.imshow(image)
        utils.draw_labels(ax, names, width, height, cell_width, cell_height, *[l[b] for l in batch_labels])
        ax.set_xticks(np.arange(0, width, width / cell_width))
        ax.set_yticks(np.arange(0, height, height / cell_height))
        ax.grid(which='both')
        ax.tick_params(labelbottom='off', labelleft='off')
    fig.tight_layout()
    plt.show()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.ini', help='config file')
    parser.add_argument('-l', '--level', default='info', help='logging level')
    parser.add_argument('-t', '--types', default=['train', 'val'])
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
