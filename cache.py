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
import pandas as pd
import voc
import utils


def main():
    section = config.get('config', 'model')
    yolo = importlib.import_module('model.' + section)
    with open(os.path.expanduser(os.path.expandvars(config.get(section, 'names'))), 'r') as f:
        names = [line.strip() for line in f]
    path = os.path.expanduser(os.path.expandvars(args.path))
    print('loading dataset from ' + path)
    imagenames, imageshapes, labels = voc.load_dataset(path, names)
    width = config.getint(section, 'width')
    height = config.getint(section, 'height')
    layers_conv = pd.read_csv(os.path.expanduser(os.path.expandvars(config.get(section, 'conv'))), sep='\t')
    cell_width = utils.calc_pooled_size(width, layers_conv['pooling1'].values)
    cell_height = utils.calc_pooled_size(height, layers_conv['pooling2'].values)
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
