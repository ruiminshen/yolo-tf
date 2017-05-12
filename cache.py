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
import shutil
import importlib
import pandas as pd
import tensorflow as tf
import utils


def main():
    cachedir = utils.get_cachedir(config)
    os.makedirs(cachedir, exist_ok=True)
    path = os.path.join(cachedir, 'names')
    shutil.copyfile(os.path.expanduser(os.path.expandvars(config.get('cache', 'names'))), path)
    with open(path, 'r') as f:
        names = [line.strip() for line in f]
    name_index = dict([(name, i) for i, name in enumerate(names)])
    datasets = [(os.path.basename(os.path.splitext(path)[0]), pd.read_csv(os.path.expanduser(os.path.expandvars(path)), sep='\t')) for path in config.get('cache', 'datasets').split(':')]
    module = importlib.import_module('utils.data.cache')
    for profile in args.profile:
        path = os.path.join(cachedir, profile + '.tfrecord')
        tf.logging.info('write tfrecords file: ' + path)
        with tf.python_io.TFRecordWriter(path) as writer:
            for name, dataset in datasets:
                tf.logging.info('loading %s %s dataset' % (name, profile))
                func = getattr(module, name)
                for i, item in dataset.iterrows():
                    tf.logging.info('loading item %d\n%s' % (i, str(item)))
                    func(writer, name_index, profile, item, args.verify)
    tf.logging.info('%s data are saved into %s' % (str(args.profile), cachedir))
    


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', default=['config.ini'], help='config file')
    parser.add_argument('-p', '--profile', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('-v', '--verify', action='store_true')
    parser.add_argument('--level', default='info', help='logging level')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    config = configparser.ConfigParser()
    utils.load_config(config, args.config)
    if args.level:
        tf.logging.set_verbosity(args.level.upper())
    main()
