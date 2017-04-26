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
import tensorflow as tf
import voc


def main():
    section = config.get('config', 'model')
    yolo = importlib.import_module(section)
    with open(os.path.expanduser(os.path.expandvars(config.get(section, 'names'))), 'r') as f:
        names = [line.strip() for line in f]
    path = os.path.expanduser(os.path.expandvars(args.path))
    print('loading dataset from ' + path)
    imagenames, imageshapes, labels = voc.load_dataset(path, names)
    width = config.getint(section, 'width')
    height = config.getint(section, 'height')
    inference = getattr(getattr(yolo, 'inference'), config.get(section, 'inference'))
    _scope, net = inference(tf.zeros([1, height, width, 3]), len(names), 1)
    try:
        _, cell_height, cell_width, _ = net.get_shape().as_list()
    except ValueError:
        _, cell_height, cell_width, _ = tf.get_default_graph().get_tensor_by_name("%s/conv:0" % _scope).get_shape().as_list()
    print('size=%d, (width, height)=(%d, %d), (cell_width, cell_height)=(%d, %d)' % (len(imagenames), width, height, cell_width, cell_height))
    labels = yolo.transform_labels_voc(imageshapes, labels, width, height, cell_width, cell_height, len(names))
    imagepaths = [os.path.join(path, 'JPEGImages', name) for name in imagenames]
    path = os.path.expanduser(os.path.expandvars(config.get(section, 'cache')))
    with open(path, 'wb') as f:
        pickle.dump((imagepaths, *labels), f)
    print('cache saved into ' + path)


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='PASCAL VOC data directory')
    parser.add_argument('-c', '--config', default='config.ini', help='config file')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    assert os.path.exists(args.config)
    config.read(args.config)
    main()
