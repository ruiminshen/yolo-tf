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
import tensorflow as tf
from tensorflow.python.framework import ops
import utils


def main():
    section = config.get('config', 'model')
    yolo = importlib.import_module('model.' + section)
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
    path = os.path.expanduser(os.path.expandvars(config.get(section, 'cache')))
    logger.info('loading cache from ' + path)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    logger.info('size: %d (batch size: %d)' % (len(data[0]), args.batch_size))
    width = config.getint(section, 'width')
    height = config.getint(section, 'height')
    with tf.Session() as sess:
        with tf.name_scope('data'):
            with tf.device('/cpu:0'):
                imagepaths = tf.train.string_input_producer(data[0], shuffle=False)
                reader = tf.WholeFileReader()
                _, image = reader.read(imagepaths)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.image.resize_images(image, [height, width])
                labels = [ops.convert_to_tensor(l, dtype=tf.float32) for l in data[1:]]
                labels = tf.train.slice_input_producer(labels, shuffle=False)
                image, labels = utils.data_augmentation(image, labels, config)
                data = tf.train.shuffle_batch([image] + labels, batch_size=args.batch_size, capacity=config.getint('queue', 'capacity'), min_after_dequeue=config.getint('queue', 'min_after_dequeue'), num_threads=multiprocessing.cpu_count())
        modeler = yolo.Modeler(args, config)
        modeler.param()
        modeler.train(data[0], data[1:])
        modeler.setup_histogram()
        with tf.name_scope('optimizer'):
            global_step = tf.Variable(0, name='global_step')
            logger.info('learning rate=%f' % args.learning_rate)
            optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(modeler.loss, global_step=global_step)
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
        os.makedirs(modeldir, exist_ok=True)
        cmd = 'tensorboard --logdir ' + logdir
        logger.info('run: ' + cmd)
        try:
            step = sess.run(global_step)
            while args.terminate <= 0 or step < args.terminate:
                _, step = sess.run([optimizer, global_step])
                if step % args.output_freq == 0:
                    logger.info('step=%d/%d' % (step, args.terminate))
                    summary_writer.add_summary(sess.run(summary), step)
                if step % args.save_freq == 0:
                    saver.save(sess, modelpath)
                    logger.info('model saved into: ' + modelpath)
        except KeyboardInterrupt:
            logger.warn('keyboard interrupt captured')
        coord.request_stop()
        coord.join(threads)
        saver.save(sess, modelpath)
        logger.info('model saved into: ' + modelpath)
        modeler.log_hparam(sess, logger)
    #os.system(cmd)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.ini', help='config file')
    parser.add_argument('-l', '--level', default='info', help='logging level')
    parser.add_argument('-t', '--terminate', type=int, default=0, help='terminate steps')
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
