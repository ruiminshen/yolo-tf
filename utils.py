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
import math
import sympy


def load(sess, saver, path_model, logger):
    modeldir = os.path.dirname(path_model)
    try:
        if os.path.exists(modeldir):
            saver.restore(sess, path_model)
            logger.info('load model: ' + path_model)
        else:
            logger.warn('modeldir %s not exists' % modeldir)
    except:
        logger.warn('failed to load model: ' + path_model)


def save(sess, saver, path_model, logger):
    os.makedirs(os.path.dirname(path_model), exist_ok=True)
    saver.save(sess, path_model)
    logger.info('model saved into: ' + path_model)


def get_factor2(x):
    factors = sympy.divisors(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]
    else:
        i = len(factors) // 2
        return factors[i], factors[i]


def calc_pooled_size(size, pooling):
    for ksize in pooling:
        if ksize > 0:
            size = math.ceil(float(size) / ksize)
    return size
