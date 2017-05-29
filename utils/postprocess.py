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

import numpy as np


def iou(xy_min1, xy_max1, xy_min2, xy_max2):
    assert(not np.isnan(xy_min1).any())
    assert(not np.isnan(xy_max1).any())
    assert(not np.isnan(xy_min2).any())
    assert(not np.isnan(xy_max2).any())
    assert np.all(xy_min1 <= xy_max1)
    assert np.all(xy_min2 <= xy_max2)
    areas1 = np.multiply.reduce(xy_max1 - xy_min1)
    areas2 = np.multiply.reduce(xy_max2 - xy_min2)
    _xy_min = np.maximum(xy_min1, xy_min2) 
    _xy_max = np.minimum(xy_max1, xy_max2)
    _wh = np.maximum(_xy_max - _xy_min, 0)
    _areas = np.multiply.reduce(_wh)
    assert _areas <= areas1
    assert _areas <= areas2
    return _areas / np.maximum(areas1 + areas2 - _areas, 1e-10)


def non_max_suppress(conf, xy_min, xy_max, threshold, threshold_iou):
    _, _, classes = conf.shape
    boxes = [(_conf, _xy_min, _xy_max) for _conf, _xy_min, _xy_max in zip(conf.reshape(-1, classes), xy_min.reshape(-1, 2), xy_max.reshape(-1, 2))]
    for c in range(classes):
        boxes.sort(key=lambda box: box[0][c], reverse=True)
        for i in range(len(boxes) - 1):
            box = boxes[i]
            if box[0][c] <= threshold:
                continue
            for _box in boxes[i + 1:]:
                if iou(box[1], box[2], _box[1], _box[2]) >= threshold_iou:
                    _box[0][c] = 0
    return boxes
