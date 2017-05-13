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
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import utils.data
import utils.visualize


def main():
    model = config.get('config', 'model')
    cachedir = utils.get_cachedir(config)
    with open(os.path.join(cachedir, 'names'), 'r') as f:
        names = [line.strip() for line in f]
    width = config.getint(model, 'width')
    height = config.getint(model, 'height')
    cell_width, cell_height = utils.calc_cell_width_height(config, width, height)
    tf.logging.info('(width, height)=(%d, %d), (cell_width, cell_height)=(%d, %d)' % (width, height, cell_width, cell_height))
    batch_size = args.rows * args.cols
    with tf.Session() as sess:
        with tf.name_scope('batch'):
            image_rgb, labels = utils.data.load_image_labels([os.path.join(cachedir, profile + '.tfrecord') for profile in args.profile], len(names), width, height, cell_width, cell_height, config)
            batch = tf.train.shuffle_batch((tf.cast(image_rgb, tf.uint8),) + labels, batch_size=batch_size,
                capacity=config.getint('queue', 'capacity'), min_after_dequeue=config.getint('queue', 'min_after_dequeue'), num_threads=multiprocessing.cpu_count()
            )
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        batch_image, batch_labels = sess.run([batch[0], batch[1:]])
        coord.request_stop()
        coord.join(threads)
    batch_image = batch_image.astype(np.uint8)
    fig, axes = plt.subplots(args.rows, args.cols)
    for b, (ax, image) in enumerate(zip(axes.flat, batch_image)):
        ax.imshow(image)
        utils.visualize.draw_labels(ax, names, width, height, cell_width, cell_height, *[l[b] for l in batch_labels])
        if args.grid:
            ax.set_xticks(np.arange(0, width, width / cell_width))
            ax.set_yticks(np.arange(0, height, height / cell_height))
            ax.grid(which='both')
            ax.tick_params(labelbottom='off', labelleft='off')
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    plt.show()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-p', '--profile', nargs='+', default=['train', 'val'])
    parser.add_argument('-g', '--grid', action='store_true')
    parser.add_argument('--rows', default=5, type=int)
    parser.add_argument('--cols', default=5, type=int)
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
