import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import os
import numpy as np
    
import colors


class PlaneLayer(object):

    def __init__(self, color, width=10, height=150):
        self.color = color
        self.width = width
        self.height = height

    def _patch(self, points, lw):
        codes = [Path.MOVETO]
        codes += [Path.LINETO] * (len(points) - 1)
        codes += [Path.CLOSEPOLY]
        path = Path(points + [points[0]], codes)
        patch = patches.PathPatch(path, facecolor=self.color, lw=lw)
        return patch

    def draw(self, lw=4, figsize=(10, 10), downsample_factor=1.0):    
        w = 0.25 * self.height * downsample_factor + self.width
        y_lim = 1.6 * self.height
        h =  y_lim * downsample_factor
        
        height = downsample_factor * self.height
        figures = [
            [(0, 0), (self.width, 0), (self.width, height), (0, height)],
            [(0, height), (self.width, height), (w, h), (w - self.width, h)],
            [(self.width, 0), (w, h - height), (w, h), (self.width, height)]
        ]
        patches = [self._patch(f, lw) for f in figures]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for patch in patches:
            ax.add_patch(patch)
        ax.set_xlim(- w * 0.05, w * 1.05)
        ax.set_ylim(- y_lim * 0.05, y_lim * 1.05)
        plt.tight_layout()
        plt.axes().set_aspect('equal')
        plt.axis('off')
        fig.canvas.draw()
        
        # To array
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        mask = data == 255
        mask = mask.astype(np.uint8)
        mask = 1 - np.prod(mask, 2)
        mask *= 255
        mask = np.expand_dims(mask, 2)
        
        # crop
        x_min = np.where(mask > 0)[1].min()
        x_min = max(x_min - 20, 0)
        x_max = np.where(mask > 0)[1].max()
        x_max = min(x_max + 20, mask.shape[1] - 1)

        y_min = np.where(mask > 0)[0].min()
        y_min = max(y_min - 20, 0)
        y_max = np.where(mask > 0)[0].max()
        y_max = min(y_max + 20, mask.shape[0] - 1)

        return np.dstack((data, mask)).astype(np.uint8)[y_min:y_max, x_min:x_max, :]


class InputLayer(PlaneLayer):

    def __init__(self, input_size=(128, 128)):
        color = colors.light_grey
        super(InputLayer, self).__init__(color=color)
        self.input_size = input_size


class Conv2D(PlaneLayer):

    def __init__(self, num_filters=32, filter_size=(1,1)):
        color = colors.blue
        super(Conv2D, self).__init__(color=color)
        self.num_filters = num_filters
        self.filter_size = filter_size


class Upsampling(PlaneLayer):

    def __init__(self, factor=None):
        color = colors.red
        self.factor = factor
        super(Upsampling, self).__init__(color=color)


class Dropout(PlaneLayer):

    def __init__(self, p=None):
        color = colors.navy
        super(Dropout, self).__init__(color=color)
        self.p = p


class Pooling(PlaneLayer):

    def __init__(self, pool_size=1):
        color = colors.green
        super(Pooling, self).__init__(color=color)
        self.pool_size = pool_size



