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
import shutil
import time
import multiprocessing
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
import yolo as model
import utils


def output(sess, step, summary, summary_writer, saver, path_model):
    if step % args.output_cycle == 0:
        evaluation = step * args.batch_size
        logger.info('evaluation=%d/%d' % (evaluation, args.evaluation))
        summary_writer.add_summary(sess.run(summary), step)
    if step % args.save_cycle == 0:
        os.makedirs(os.path.dirname(path_model), exist_ok=True)
        saver.save(sess, path_model)
        logger.info('model saved into: ' + path_model)


def main():
    yolodir = os.path.expanduser(os.path.expandvars(config.get('yolo', 'dir')))
    modeldir = os.path.join(yolodir, 'model')
    path_model = os.path.join(modeldir, 'model.ckpt')
    if args.reset and os.path.exists(modeldir):
        logger.warn('delete model_train: ' + modeldir)
        shutil.rmtree(modeldir, ignore_errors=True)
    logdir = os.path.join(yolodir, 'logdir')
    if args.delete:
        logger.warn('delete logdir: ' + logdir)
        shutil.rmtree(logdir, ignore_errors=True)
    path = os.path.expanduser(os.path.expandvars(config.get(model.__name__, 'cache')))
    logger.info('loading cache from ' + path)
    with open(path, 'rb') as f:
        names = pickle.load(f)
        data = pickle.load(f)
    logger.info('size: %d (batch size: %d)' % (len(data[0]), args.batch_size))
    width = config.getint(model.__name__, 'width')
    height = config.getint(model.__name__, 'height')
    layers_conv = pd.read_csv(os.path.expanduser(os.path.expandvars(config.get(model.__name__, 'conv'))), sep='\t')
    cell_width = utils.calc_pooled_size(width, layers_conv['pooling1'].values)
    cell_height = utils.calc_pooled_size(height, layers_conv['pooling2'].values)
    layers_fc = pd.read_csv(os.path.expanduser(os.path.expandvars(config.get(model.__name__, 'fc'))), sep='\t')
    boxes_per_cell = config.getint(model.__name__, 'boxes_per_cell')
    with tf.Session() as sess:
        logger.info('init param')
        with tf.variable_scope('param'):
            param_conv = model.ParamConv(3, layers_conv, seed=args.seed)
            inputs = cell_width * cell_height * param_conv.bais[-1].get_shape()[0].value
            outputs = cell_width * cell_height * (len(names) + boxes_per_cell * 5)
            param_fc = model.ParamFC(inputs, layers_fc, outputs, seed=args.seed)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        with tf.name_scope('data'):
            with tf.device('/cpu:0'):
                imagepaths = tf.train.string_input_producer(data[0], shuffle=False)
                reader = tf.WholeFileReader()
                _, image = reader.read(imagepaths)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.image.resize_images(image, [height, width])
                image = tf.image.per_image_standardization(image)
                labels = [ops.convert_to_tensor(l, dtype=tf.float32) for l in data[1:]]
                labels = tf.train.slice_input_producer(labels, shuffle=False)
                data = tf.train.shuffle_batch([image] + labels, batch_size=args.batch_size, capacity=args.batch_size * config.getint('queue', 'capacity'), min_after_dequeue=args.batch_size * config.getint('queue', 'min'), num_threads=multiprocessing.cpu_count())
        logger.info('init model')
        with tf.name_scope('train'):
            model_train = model.Model(data[0], param_conv, param_fc, layers_conv, layers_fc, len(names), boxes_per_cell, train=True, seed=args.seed)
        with tf.name_scope('loss'):
            loss_train = model.Loss(model_train, *data[1:])
            with tf.variable_scope('hparam'):
                hparam = dict([(key, tf.Variable(float(s), name='hparam_' + key, trainable=False)) for key, s in config.items(model.__name__ + '_hparam')])
                hparam_regularizer = tf.Variable(config.getfloat(model.__name__, 'hparam'), name='hparam_regularizer', trainable=False)
            loss = tf.reduce_sum([loss_train[key] * hparam[key] for key in loss_train], name='loss_objectives') + tf.multiply(hparam_regularizer, model_train.regularizer, name='loss_regularizer')
            for key in loss_train:
                tf.summary.scalar(key, loss_train[key])
            tf.summary.scalar('regularizer', model_train.regularizer)
            tf.summary.scalar('loss', loss)
        with tf.name_scope('optimizer'):
            step = tf.Variable(0, name='step')
            logger.info('learning rate=%f' % args.learning_rate)
            optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=step)
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(logdir, time.strftime('%Y-%m-%d_%H-%M-%S')), sess.graph)
        tf.global_variables_initializer().run()
        cmd = 'tensorboard --logdir ' + logdir
        logger.info('run: ' + cmd)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        logger.info('load model')
        saver = tf.train.Saver()
        try:
            saver.restore(sess, path_model)
        except:
            logger.warn('error occurs while loading model: ' + path_model)
        logger.info(', '.join(['%s=%f' % (key, p) for key, p in zip(hparam.keys(), sess.run([hparam[key] for key in hparam]))]))
        logger.info('hparam_regularizer=%f' % sess.run(hparam_regularizer))
        try:
            _step = sess.run(step)
            while args.evaluation <= 0 or _step * args.batch_size < args.evaluation:
                _, _step = sess.run([optimizer, step])
                output(sess, _step, summary, summary_writer, saver, path_model)
        except KeyboardInterrupt:
            logger.warn('keyboard interrupt captured')
        coord.request_stop()
        coord.join(threads)
        logger.info('save model')
        os.makedirs(os.path.dirname(path_model), exist_ok=True)
        saver.save(sess, path_model)
        logger.info('model saved into: ' + path_model)
        logger.info(', '.join(['%s=%f' % (key, p) for key, p in zip(hparam.keys(), sess.run([hparam[key] for key in hparam]))]))
        logger.info('hparam_regularizer=%f' % sess.run(hparam_regularizer))
    #os.system(cmd)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.ini', help='config file')
    parser.add_argument('-l', '--level', default='info', help='logging level')
    parser.add_argument('-e', '--evaluation', type=int, default=0)
    parser.add_argument('-r', '--reset', action='store_true', help='reset saved model')
    parser.add_argument('-d', '--delete', action='store_true', help='delete logdir')
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-b', '--batch_size', default=16, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output_cycle', default=10, type=int)
    parser.add_argument('--save_cycle', default=500, type=int)
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
