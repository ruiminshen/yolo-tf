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
import inspect
import multiprocessing
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils.data


def summary_scalar(config):
    try:
        reduce = eval(config.get('summary', 'scalar_reduce'))
        for t in utils.match_tensor(config.get('summary', 'scalar')):
            name = t.op.name
            if len(t.get_shape()) > 0:
                t = reduce(t)
                tf.logging.warn(name + ' is not a scalar tensor, reducing by ' + reduce.__name__)
            tf.summary.scalar(name, t)
    except (configparser.NoSectionError, configparser.NoOptionError):
        tf.logging.warn(inspect.stack()[0][3] + ' disabled')


def summary_image(config):
    try:
        for t in utils.match_tensor(config.get('summary', 'image')):
            name = t.op.name
            channels = t.get_shape()[-1].value
            if channels not in (1, 3, 4):
                t = tf.expand_dims(tf.reduce_sum(t, -1), -1)
            tf.summary.image(name, t, config.getint('summary', 'image_max'))
    except (configparser.NoSectionError, configparser.NoOptionError):
        tf.logging.warn(inspect.stack()[0][3] + ' disabled')


def summary_histogram(config):
    try:
        for t in utils.match_tensor(config.get('summary', 'histogram')):
            tf.summary.histogram(t.op.name, t)
    except (configparser.NoSectionError, configparser.NoOptionError):
        tf.logging.warn(inspect.stack()[0][3] + ' disabled')


def summary(config):
    summary_scalar(config)
    summary_image(config)
    summary_histogram(config)


__optimizers__ = {
    'adam': lambda learning_rate: tf.train.AdamOptimizer(learning_rate),
    'adadelta': lambda learning_rate: tf.train.AdadeltaOptimizer(learning_rate),
    'adagrad': lambda learning_rate: tf.train.AdagradOptimizer(learning_rate),
    'momentum': lambda learning_rate: tf.train.MomentumOptimizer(learning_rate, 0.9),
    'rmsprop': lambda learning_rate: tf.train.RMSPropOptimizer(learning_rate),
    'ftrl': lambda learning_rate: tf.train.FtrlOptimizer(learning_rate),
    'gd': lambda learning_rate: tf.train.GradientDescentOptimizer(learning_rate),
}


def get_optimizer(config, name):
    section = 'optimizer_' + name
    return {
        'adam': lambda learning_rate: tf.train.AdamOptimizer(learning_rate),
        'momentum': lambda learning_rate: tf.train.MomentumOptimizer(learning_rate, config.getfloat(section, 'momentum')),
        'rmsprop': lambda learning_rate: tf.train.RMSPropOptimizer(learning_rate),
    }[name]


def main():
    model = config.get('config', 'model')
    logdir = utils.get_logdir(config)
    if args.delete:
        tf.logging.warn('delete logging directory: ' + logdir)
        shutil.rmtree(logdir, ignore_errors=True)
    cachedir = utils.get_cachedir(config)
    with open(os.path.join(cachedir, 'names'), 'r') as f:
        names = [line.strip() for line in f]
    width = config.getint(model, 'width')
    height = config.getint(model, 'height')
    cell_width, cell_height = utils.calc_cell_width_height(config, width, height)
    tf.logging.warn('(width, height)=(%d, %d), (cell_width, cell_height)=(%d, %d)' % (width, height, cell_width, cell_height))
    yolo = importlib.import_module('model.' + model)
    with tf.name_scope('batch'):
        image_rgb, labels = utils.data.load_image_labels([os.path.join(cachedir, profile + '.tfrecord') for profile in args.profile], len(names), width, height, cell_width, cell_height, config)
        with tf.name_scope('per_image_standardization'):
            image_std = tf.image.per_image_standardization(image_rgb)
        batch = tf.train.shuffle_batch((image_std,) + labels, batch_size=args.batch_size,
            capacity=config.getint('queue', 'capacity'), min_after_dequeue=config.getint('queue', 'min_after_dequeue'),
            num_threads=multiprocessing.cpu_count()
        )
    global_step = tf.contrib.framework.get_or_create_global_step()
    builder = yolo.Builder(args, config)
    builder(batch[0], training=True)
    variables_to_restore = slim.get_variables_to_restore(exclude=args.exclude)
    loss = builder.loss(batch[1:])
    with tf.name_scope('optimizer'):
        try:
            decay_steps = config.getint('exponential_decay', 'decay_steps')
            decay_rate = config.getfloat('exponential_decay', 'decay_rate')
            staircase = config.getboolean('exponential_decay', 'staircase')
            learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, decay_steps, decay_rate, staircase=staircase)
            tf.logging.warn('using a learning rate start from %f with exponential decay (decay_steps=%d, decay_rate=%f, staircase=%d)' % (args.learning_rate, decay_steps, decay_rate, staircase))
        except (configparser.NoSectionError, configparser.NoOptionError):
            learning_rate = args.learning_rate
            tf.logging.warn('using a staionary learning rate %f' % args.learning_rate)
        optimizer = get_optimizer(config, args.optimizer)(learning_rate)
        tf.logging.warn('optimizer=' + args.optimizer)
        train_op = slim.learning.create_train_op(loss, optimizer, global_step, clip_gradient_norm=args.clip_gradient)
    if args.finetune:
        path = os.path.expanduser(os.path.expandvars(args.finetune))
        tf.logging.warn('fine-tuning from ' + path)
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(path, variables_to_restore)
        def init_fn(sess):
            sess.run(init_assign_op, init_feed_dict)
            tf.logging.warn('fine-tuning from global_step=%d, learning_rate=%f' % sess.run((global_step, learning_rate)))
    else:
        init_fn = lambda sess: tf.logging.warn('global_step=%d, learning_rate=%f' % sess.run((global_step, learning_rate)))
    summary(config)
    tf.logging.warn('tensorboard --logdir ' + logdir)
    slim.learning.train(train_op, logdir, master=args.master, is_chief=(args.task == 0), global_step=global_step, number_of_steps=args.steps, init_fn=init_fn,
        summary_writer=tf.summary.FileWriter(os.path.join(logdir, args.logname)),
        save_summaries_secs=args.summary_secs, save_interval_secs=args.save_secs
    )


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-f', '--finetune', help='initializing model from a .ckpt file')
    parser.add_argument('-e', '--exclude', nargs='+', help='exclude variables while fine-tuning')
    parser.add_argument('-p', '--profile', nargs='+', default=['train', 'val'])
    parser.add_argument('-m', '--master', default='', help='master address')
    parser.add_argument('-t', '--task', type=int, default=0, help='task ID')
    parser.add_argument('-s', '--steps', type=int, default=None, help='max number of steps')
    parser.add_argument('-d', '--delete', action='store_true', help='delete logdir')
    parser.add_argument('-b', '--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('-o', '--optimizer', default='adam')
    parser.add_argument('-n', '--logname', default=time.strftime('%Y-%m-%d_%H-%M-%S'), help='the name for TensorBoard')
    parser.add_argument('-lr', '--learning_rate', default=1e-5, type=float, help='learning rate')
    parser.add_argument('-cg', '--clip_gradient', default=0, type=int, help='clip gradient')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--summary_secs', default=30, type=int, help='seconds to save summaries')
    parser.add_argument('--save_secs', default=600, type=int, help='seconds to save model')
    parser.add_argument('--level', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
