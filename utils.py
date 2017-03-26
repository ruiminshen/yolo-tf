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

import sys
import math
import logging
import getpass
import sympy


def make_logger(level, fmt):
    logger = logging.getLogger(getpass.getuser())
    logger.setLevel(level)
    formatter = logging.Formatter(fmt)
    settings = [
        (logging.INFO, sys.stdout),
        (logging.WARN, sys.stderr),
    ]
    for level, out in settings:
        handler = logging.StreamHandler(out)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


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
