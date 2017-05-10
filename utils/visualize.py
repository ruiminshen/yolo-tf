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

import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_labels(ax, names, width, height, cell_width, cell_height, mask, prob, coords, xy_min, xy_max, areas, color='red', rtol=1e-3):
    colors = [prop['color'] for _, prop in zip(names, itertools.cycle(plt.rcParams['axes.prop_cycle']))]
    plots = []
    for i, (_mask, _prob, _coords, _xy_min, _xy_max, _areas) in enumerate(zip(mask, prob, coords, xy_min, xy_max, areas)):
        _mask = _mask.reshape([])
        _coords = _coords.reshape([-1])
        if np.any(_mask) > 0:
            iy = i // cell_width
            ix = i % cell_width
            plots.append(ax.add_patch(patches.Rectangle((ix * width / cell_width, iy * height / cell_height), width / cell_width, height / cell_height, linewidth=0, facecolor=color, alpha=.2)))
            #check coords
            offset_x, offset_y, _w_sqrt, _h_sqrt = _coords
            cell_x, cell_y = ix + offset_x, iy + offset_y
            x, y = cell_x * width / cell_width, cell_y * height / cell_height
            _w, _h = _w_sqrt * _w_sqrt, _h_sqrt * _h_sqrt
            w, h = _w * width, _h * height
            x_min, y_min = x - w / 2, y - h / 2
            index = np.argmax(_prob)
            plots.append(ax.add_patch(patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor=colors[index], facecolor='none')))
            plots.append(ax.annotate(names[index], (x_min, y_min), color=colors[index]))
            #check offset_xy_min and xy_max
            wh = _xy_max - _xy_min
            assert np.all(wh >= 0)
            np.testing.assert_allclose(wh / [cell_width, cell_height], [[_w, _h]], rtol=rtol)
            np.testing.assert_allclose(_xy_min + wh / 2, [[offset_x, offset_y]], rtol=rtol)
    return plots
