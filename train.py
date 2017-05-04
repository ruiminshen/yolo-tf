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
import shutil
import time
import multiprocessing
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils


def summary_scalar(builder):
    for key in builder.objectives:
        tf.summary.scalar(key, builder.objectives[key])
    try:
        tf.summary.scalar('regularizer', builder.model.regularizer)
    except AttributeError:
        logger.warn('model regularizer not exists')


def tensorboard_histogram(config):
    try:
        for t in utils.match_tensor(config.get('tensorboard', 'histogram')):
            tf.summary.histogram(t.op.name, t)
    except configparser.NoOptionError:
        logger.warn('no option histogram in section tensorboard')


def log_hparam(builder, sess):
    keys, values = zip(*builder.hparam.items())
    logger.info(', '.join(['%s=%f' % (key, value) for key, value in zip(keys, sess.run(values))]))
    try:
        logger.info('hparam_regularizer=%f' % sess.run(builder.hparam_regularizer))
    except AttributeError:
        pass


__optimizers__ = {
    'adam': lambda learning_rate: tf.train.AdamOptimizer(learning_rate),
    'momentum': lambda learning_rate: tf.train.MomentumOptimizer(learning_rate, 0.9),
    'rmsprop': lambda learning_rate: tf.train.RMSPropOptimizer(learning_rate),
}


def main():
    model = config.get('config', 'model')
    logdir = utils.get_logdir(config)
    if args.delete:
        logger.warn('delete logging directory: ' + logdir)
        shutil.rmtree(logdir, ignore_errors=True)
    cachedir = utils.get_cachedir(config)
    with open(os.path.join(cachedir, 'names'), 'r') as f:
        names = [line.strip() for line in f]
    width = config.getint(model, 'width')
    height = config.getint(model, 'height')
    yolo = importlib.import_module(model)
    downsampling = utils.get_downsampling(config)
    assert width % downsampling == 0
    assert height % downsampling == 0
    cell_width, cell_height = width // downsampling, height // downsampling
    logger.info('(width, height)=(%d, %d), (cell_width, cell_height)=(%d, %d)' % (width, height, cell_width, cell_height))
    with tf.name_scope('batch'):
        image_rgb, labels = utils.load_image_labels([os.path.join(cachedir, profile + '.tfrecord') for profile in args.profile], len(names), width, height, cell_width, cell_height, config)
        with tf.name_scope('per_image_standardization'):
            image_std = tf.image.per_image_standardization(image_rgb)
        batch = tf.train.shuffle_batch((image_std,) + labels, batch_size=args.batch_size,
            capacity=config.getint('queue', 'capacity'), min_after_dequeue=config.getint('queue', 'min_after_dequeue'), num_threads=multiprocessing.cpu_count()
        )
    builder = yolo.Builder(args, config)
    builder(batch[0], training=True)
    loss = builder.loss(batch[1:])
    with tf.name_scope('loss'):
        summary_scalar(builder)
        tf.summary.scalar('loss', loss)
    tensorboard_histogram(config)
    logger.info('optimizer=%s, learning rate=%f' % (args.optimizer, args.learning_rate))
    with tf.name_scope('optimizer'):
        optimizer = __optimizers__[args.optimizer](args.learning_rate)
        train_op = slim.learning.create_train_op(loss, optimizer)
    with tf.name_scope('train'):
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(os.path.join(logdir, args.logname))
        logger.info('tensorboard --logdir ' + logdir)
        slim.learning.train(train_op, logdir,
            master=args.master, is_chief=(args.task == 0),
            saver=saver, summary_writer=summary_writer,
            number_of_steps=args.steps,
            save_summaries_secs=args.summary_secs, save_interval_secs=args.save_secs
        )


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.ini', help='config file')
    parser.add_argument('-l', '--level', default='info', help='logging level')
    parser.add_argument('-p', '--profile', nargs='+', default=['train', 'val'])
    parser.add_argument('-m', '--master', default='', help='master address')
    parser.add_argument('-t', '--task', type=int, default=0, help='task ID')
    parser.add_argument('-s', '--steps', type=int, default=None, help='max number of steps')
    parser.add_argument('-d', '--delete', action='store_true', help='delete logdir')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('-o', '--optimizer', default='adam')
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--summary_secs', default=5, type=int, help='seconds to save summaries')
    parser.add_argument('--save_secs', default=600, type=int, help='seconds to save model')
    parser.add_argument('--logname', default=time.strftime('%Y-%m-%d_%H-%M-%S'), help='the name of TensorBoard log')
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
