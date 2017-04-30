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
import matplotlib.patches as patches
import tensorflow as tf
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
        height, width, _ = image.shape
        vmin = np.min(image)
        vmax = np.max(image)
        image = ((image - vmin) * 255 / (vmax - vmin)).astype(np.uint8)
        self.ax.imshow(image)
        self.plots = utils.draw_labels(self.ax, names, width, height, cell_width, cell_height, *labels)
        self.ax.set_xticks(np.arange(0, width, width / cell_width))
        self.ax.set_yticks(np.arange(0, height, height / cell_height))
        self.ax.grid(which='both')
        self.ax.tick_params(labelbottom='off', labelleft='off')
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
        prob, iou, xy_min, wh = self.sess.run([self.model.prob[0][index], self.model.iou[0][index], self.model.xy_min[0][index], self.model.wh[0][index]])
        xy_min = xy_min * [image_width, image_height] / [self.cell_width, self.cell_height]
        wh = wh * [image_width, image_height] / [self.cell_width, self.cell_height]
        for _prob, _iou, (x, y), (w, h), color in zip(prob, iou, xy_min, wh, self.colors):
            index = np.argmax(_prob)
            name = self.names[index]
            _prob = _prob[index]
            _conf = _prob * _iou
            linewidth = min(_conf * 10, 3)
            self.plots.append(self.ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=linewidth, edgecolor=color, facecolor='none')))
            self.plots.append(self.ax.annotate(name + ' (%.1f%%, %.1f%%)' % (_iou * 100, _prob * 100), (x, y), color=color))
        self.fig.canvas.draw()


def main():
    section = config.get('config', 'model')
    yolo = importlib.import_module(section)
    basedir = os.path.expanduser(os.path.expandvars(config.get(section, 'basedir')))
    modeldir = os.path.join(basedir, 'model')
    modelpath = os.path.join(modeldir, 'model.ckpt')
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
            _, serialized = reader.read(tf.train.string_input_producer([os.path.join(cachedir, profile + '.tfrecord') for profile in args.profile], shuffle=False))
            example = tf.parse_single_example(serialized, features={
                'imagepath': tf.FixedLenFeature([], tf.string),
                'objects': tf.FixedLenFeature([2], tf.string),
            })
            image_rgb, objects_class, objects_coord = utils.decode_image_objects(example, width, height)
            if config.getboolean('data_augmentation', 'enable'):
                image_rgb, objects_coord = utils.data_augmentation(image_rgb, objects_coord, config)
            image_std = tf.image.per_image_standardization(image_rgb)
            labels = utils.decode_labels(objects_class, objects_coord, len(names), cell_width, cell_height)
            batch = tf.train.shuffle_batch((image_std,) + labels, batch_size=1, capacity=config.getint('queue', 'capacity'), min_after_dequeue=config.getint('queue', 'min_after_dequeue'), num_threads=multiprocessing.cpu_count())
            image = tf.identity(batch[0], name='image')
        builder = yolo.Builder(args, config)
        builder(image)
        loss = builder.loss(labels)
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        batch_image, batch_labels = sess.run([image, batch[1:]])
        coord.request_stop()
        coord.join(threads)
        logger.info('load model')
        saver = tf.train.Saver()
        saver.restore(sess, modelpath)
        image, labels = batch_image[0], [l[0] for l in batch_labels]
        _ = Drawer(sess, names, builder.model.cell_width, builder.model.cell_height, image, labels, builder.model, loss)
        plt.show()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.ini', help='config file')
    parser.add_argument('-l', '--level', default='info', help='logging level')
    parser.add_argument('-p', '--profile', nargs='+', default=['train', 'val'])
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
