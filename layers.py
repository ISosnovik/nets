import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import os
import numpy as np
    
import colors



def crop_xy(array):    
    x_min = np.where(array > 0)[1].min()
    x_min = max(x_min - 20, 0)
    x_max = np.where(array > 0)[1].max()
    x_max = min(x_max + 20, array.shape[1] - 1)

    y_min = np.where(array > 0)[0].min()
    y_min = max(y_min - 20, 0)
    y_max = np.where(array > 0)[0].max()
    y_max = min(y_max + 20, array.shape[0] - 1)
    return x_min, y_min, x_max, y_max


def poly_patch(points, lw, color):
    codes = [Path.MOVETO]
    codes += [Path.LINETO] * (len(points) - 1)
    codes += [Path.CLOSEPOLY]
    path = Path(points + [points[0]], codes)
    patch = patches.PathPatch(path, facecolor=color, lw=lw)
    return patch


def figure2array(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    mask = data == 255
    mask = mask.astype(np.uint8)
    mask = 1 - np.prod(mask, 2)
    mask *= 255
    mask = np.expand_dims(mask, 2)

    array = np.dstack((data, mask)).astype(np.uint8)
    x_min, y_min, x_max, y_max = crop_xy(mask)
    return array[y_min:y_max, x_min:x_max, :]


# General Class
class PlaneLayer(object):

    def __init__(self, color):
        self.color = color
        self.width = 10
        self.height = 150

    def draw(self, lw=2, figsize=(10, 10)):  
        w = 0.25 * self.height + self.width
        h = 1.6 * self.height

        figures = [
            [(0, 0), (self.width, 0), (self.width, self.height), (0, self.height)],
            [(0, self.height), (self.width, self.height), (w, h), (w - self.width, h)],
            [(self.width, 0), (w, h - self.height), (w, h), (self.width, self.height)]
        ]
        patches = [poly_patch(f, lw, self.color) for f in figures]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for patch in patches:
            ax.add_patch(patch)
        ax.set_xlim(- 10, w + 10)
        ax.set_ylim(- 20, 260)
        plt.axes().set_aspect('equal')
        plt.axis('off')
        fig.canvas.draw()
        
        data = figure2array(fig)
        plt.close(fig)
        return data



# Basic Layers
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



