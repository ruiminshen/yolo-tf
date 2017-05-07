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
import re
import time
import shutil
import argparse
import configparser
import operator
import itertools
import struct
import numpy as np
import pandas as pd
import tensorflow as tf
import yolo2.inference as inference
import utils


def transpose_weights(weights, num_anchors):
    ksize1, ksize2, channels_in, _ = weights.shape
    weights = weights.reshape([ksize1, ksize2, channels_in, num_anchors, -1])
    coords = weights[:, :, :, :, 0:4]
    iou = np.expand_dims(weights[:, :, :, :, 4], -1)
    classes = weights[:, :, :, :, 5:]
    return np.concatenate([iou, coords, classes], -1).reshape([ksize1, ksize2, channels_in, -1])


def transpose_biases(biases, num_anchors):
    biases = biases.reshape([num_anchors, -1])
    coords = biases[:, 0:4]
    iou = np.expand_dims(biases[:, 4], -1)
    classes = biases[:, 5:]
    return np.concatenate([iou, coords, classes], -1).reshape([-1])


def transpose(sess, layer, num_anchors):
    v = next(filter(lambda v: v.op.name.endswith('weights'), layer))
    sess.run(v.assign(transpose_weights(sess.run(v), num_anchors)))
    v = next(filter(lambda v: v.op.name.endswith('biases'), layer))
    sess.run(v.assign(transpose_biases(sess.run(v), num_anchors)))


def main():
    model = config.get('config', 'model')
    cachedir = utils.get_cachedir(config)
    with open(os.path.join(cachedir, 'names'), 'r') as f:
        names = [line.strip() for line in f]
    width, height = np.array(utils.get_downsampling(config)) * 13
    anchors = pd.read_csv(os.path.expanduser(os.path.expandvars(config.get(model, 'anchors'))), sep='\t').values
    func = getattr(inference, config.get(model, 'inference'))
    with tf.Session() as sess:
        image = tf.placeholder(tf.float32, [1, height, width, 3], name='image')
        func(image, len(names), len(anchors))
        tf.contrib.framework.get_or_create_global_step()
        tf.global_variables_initializer().run()
        prog = re.compile(r'[_\w\d]+\/conv(\d*)\/(weights|biases|(BatchNorm\/(gamma|beta|moving_mean|moving_variance)))$')
        variables = [(prog.match(v.op.name).group(1), v) for v in tf.global_variables() if prog.match(v.op.name)]
        variables = sorted([[int(k) if k else -1, [v for _, v in g]] for k, g in itertools.groupby(variables, operator.itemgetter(0))], key=operator.itemgetter(0))
        assert variables[0][0] == -1
        variables[0][0] = len(variables) - 1
        variables.insert(len(variables), variables.pop(0))
        with tf.name_scope('assign'):
            with open(os.path.expanduser(os.path.expandvars(args.file)), 'rb') as f:
                major, minor, revision, seen = struct.unpack('4i', f.read(16))
                tf.logging.info('major=%d, minor=%d, revision=%d, seen=%d' % (major, minor, revision, seen))
                for i, layer in variables:
                    tf.logging.info('processing layer %d' % i)
                    total = 0
                    for suffix in ['biases', 'beta', 'gamma', 'moving_mean', 'moving_variance', 'weights']:
                        try:
                            v = next(filter(lambda v: v.op.name.endswith(suffix), layer))
                        except StopIteration:
                            continue
                        shape = v.get_shape().as_list()
                        cnt = np.multiply.reduce(shape)
                        total += cnt
                        tf.logging.info('%s: %s=%d' % (v.op.name, str(shape), cnt))
                        p = struct.unpack('%df' % cnt, f.read(4 * cnt))
                        if suffix == 'weights':
                            ksize1, ksize2, channels_in, channels_out = shape
                            p = np.reshape(p, [channels_out, channels_in, ksize1, ksize2]) # DarkNet format
                            p = np.transpose(p, [2, 3, 1, 0]) # TensorFlow format (ksize1, ksize2, channels_in, channels_out)
                        sess.run(v.assign(p))
                    tf.logging.info('%d parameters assigned' % total)
                remaining = os.fstat(f.fileno()).st_size - f.tell()
            transpose(sess, layer, len(anchors))
        saver = tf.train.Saver()
        logdir = utils.get_logdir(config)
        if args.delete:
            tf.logging.warn('delete logging directory: ' + logdir)
            shutil.rmtree(logdir, ignore_errors=True)
        os.makedirs(logdir, exist_ok=True)
        model_path = os.path.join(logdir, 'model.ckpt')
        tf.logging.info('save model into ' + model_path)
        saver.save(sess, model_path)
        if args.summary:
            path = os.path.join(logdir, args.logname)
            summary_writer = tf.summary.FileWriter(path)
            summary_writer.add_graph(sess.graph)
            tf.logging.info('tensorboard --logdir ' + logdir)
    if remaining > 0:
        tf.logging.warn('%d bytes remaining' % remaining)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='DarkNet .weights file')
    parser.add_argument('-c', '--config', default='config.ini', help='config file')
    parser.add_argument('-d', '--delete', action='store_true', help='delete logdir')
    parser.add_argument('-s', '--summary', action='store_true')
    parser.add_argument('--logname', default=time.strftime('%Y-%m-%d_%H-%M-%S'), help='the name of TensorBoard log')
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    assert os.path.exists(args.config)
    config.read(args.config)
    if args.level:
        tf.logging.set_verbosity(eval('tf.logging.' + args.level.upper()))
    main()
