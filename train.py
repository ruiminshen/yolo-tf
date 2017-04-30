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
import math
import multiprocessing
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import utils


def main():
    section = config.get('config', 'model')
    yolo = importlib.import_module(section)
    with open(os.path.expanduser(os.path.expandvars(config.get(section, 'names'))), 'r') as f:
        names = [line.strip() for line in f]
    basedir = os.path.expanduser(os.path.expandvars(config.get(section, 'basedir')))
    modeldir = os.path.join(basedir, 'model')
    modelpath = os.path.join(modeldir, 'model.ckpt')
    if args.reset and os.path.exists(modeldir):
        logger.warn('delete modeldir: ' + modeldir)
        shutil.rmtree(modeldir, ignore_errors=True)
    logdir = os.path.join(basedir, 'logdir')
    if args.delete:
        logger.warn('delete logdir: ' + logdir)
        shutil.rmtree(logdir, ignore_errors=True)
    width = config.getint(section, 'width')
    height = config.getint(section, 'height')
    downsampling = config.getint(section, 'downsampling')
    assert width % downsampling == 0
    assert height % downsampling == 0
    cell_width, cell_height = width // downsampling, height // downsampling
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
            with tf.name_scope('per_image_standardization'):
                image_std = tf.image.per_image_standardization(image_rgb)
            with tf.device('/cpu:0'):
                labels = utils.decode_labels(objects_class, objects_coord, len(names), cell_width, cell_height)
            batch = tf.train.shuffle_batch((image_std,) + labels, batch_size=args.batch_size, capacity=config.getint('queue', 'capacity'), min_after_dequeue=config.getint('queue', 'min_after_dequeue'), num_threads=multiprocessing.cpu_count())
            image = tf.identity(batch[0], name='image')
        builder = yolo.Builder(args, config)
        builder(image, training=True)
        loss = builder.loss(batch[1:])
        with tf.name_scope('loss'):
            for key in builder.objectives:
                tf.summary.scalar(key, builder.objectives[key])
            try:
                tf.summary.scalar('regularizer', builder.model.regularizer)
            except AttributeError:
                logger.warn('model regularizer not exists')
            tf.summary.scalar('loss', loss)
        try:
            for t in utils.match_tensor(config.get('tensorboard', 'histogram')):
                tf.summary.histogram(t.op.name, t)
        except configparser.NoOptionError:
            logger.warn('no option histogram in section tensorboard')
        with tf.name_scope('optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if update_ops:
                updates = tf.group(*update_ops)
                loss = control_flow_ops.with_dependencies([updates], loss, name='loss_update_ops')
            global_step = tf.Variable(0, name='global_step')
            logger.info('learning rate=%f' % args.learning_rate)
            optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step)
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(logdir, args.logname), sess.graph)
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        logger.info('load model')
        saver = tf.train.Saver()
        if os.path.exists(modeldir):
            try:
                saver.restore(sess, modelpath)
            except:
                logger.warn('error occurs while loading model: ' + modelpath)
        def save():
            if math.isnan(sess.run(loss)):
                raise FloatingPointError('a NaN loss value captured')
            os.makedirs(modeldir, exist_ok=True)
            saver.save(sess, modelpath)
            logger.info('model saved into: ' + modelpath)
        cmd = 'tensorboard --logdir ' + logdir
        logger.info('run: ' + cmd)
        try:
            step = sess.run(global_step)
            while args.steps <= 0 or step < args.steps:
                _, step = sess.run([optimizer, global_step])
                if step % args.output_freq == 0:
                    logger.info('step=%d/%d' % (step, args.steps))
                    summary_writer.add_summary(sess.run(summary), step)
                if step % args.save_freq == 0:
                    save()
        except KeyboardInterrupt:
            logger.warn('keyboard interrupt captured')
        coord.request_stop()
        coord.join(threads)
        save()
        builder.log_hparam(sess, logger)
    #os.system(cmd)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.ini', help='config file')
    parser.add_argument('-l', '--level', default='info', help='logging level')
    parser.add_argument('-p', '--profile', nargs='+', default=['train', 'val'])
    parser.add_argument('-s', '--steps', type=int, default=0, help='max number of steps')
    parser.add_argument('-r', '--reset', action='store_true', help='delete saved model')
    parser.add_argument('-d', '--delete', action='store_true', help='delete logdir')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output_freq', default=10, type=int, help='output frequency')
    parser.add_argument('--save_freq', default=500, type=int, help='save frequency')
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
