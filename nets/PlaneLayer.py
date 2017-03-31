import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
    
from utils import *
import colors
from legend import *


# General Class
class PlaneLayer(object):

    def __init__(self, color, description, 
        input_size=(128, 128), width=10, height=150):

        self.color = color
        self.width = width
        self.height = height
        self.description = description 
        self.input_size = input_size

    @property
    def legend(self):
        patch = patches.Rectangle((0,0), 1., 1., facecolor=self.color)
        return (patch, self.description)

    def output_size(self, input_size=None):
        pass

    def _main_patch(self, points, lw):
        codes = [Path.MOVETO]
        codes += [Path.LINETO] * (len(points) - 1)
        codes += [Path.CLOSEPOLY]
        path = Path(points + [points[0]], codes)
        patch = patches.PathPatch(path, facecolor=self.color, lw=lw)
        return patch

    def draw(self, lw=2, figsize=(10, 10)):  
        w = 0.25 * self.height + self.width
        h = 1.6 * self.height

        points = [(0, 0), (self.width, 0),  (w, h - self.height), 
                  (w, h), (w - self.width, h), (0, self.height)]
        patch = self._main_patch(points, lw)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # draw layer
        ax.add_patch(patch)
        ax.plot([0.5, self.width], [self.height, self.height], lw=lw, c='black')
        ax.plot([self.width, self.width], [self.height, 0.5], lw=lw, c='black')

        ax.plot([self.width, w - 0.5], [self.height, h - 1.2], lw=lw, c='black')

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

    def __init__(self, input_size=(128, 128), description='Input'):
        color = colors.light_grey
        super(InputLayer, self).__init__(color=color, description=description)
        self.input_size = input_size

    def output_size(self, input_size=None):
        return self.input_size[0]


class Conv2D(PlaneLayer):

    def __init__(self, num_filters=32, filter_size=(1,1), stride=1,
        mode='same', description='Conv'):

        color = colors.blue
        super(Conv2D, self).__init__(color=color, description=description)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        if stride != 1:
            mode = None
        self.mode = mode

    def output_size(self, input_size=None):
        if type(input_size) == tuple:
            input_size = input_size[0]
        if type(self.filter_size) == tuple:
            self.filter_size = self.filter_size[0]
        if self.mode == 'same':
            return input_size
        return input_size#(input_size - self.filter_size) / self.stride + 1


class Upsampling(PlaneLayer):

    def __init__(self, factor=None, description='Upsampling'):
        color = colors.red
        super(Upsampling, self).__init__(color=color, description=description)
        self.factor = factor

    def output_size(self, input_size=None):
        if type(input_size) == tuple:
            input_size = input_size[0]
        return self.factor * input_size


class Dropout(PlaneLayer):

    def __init__(self, p=None, description='Dropout'):
        color = colors.navy
        super(Dropout, self).__init__(color=color, description=description)
        self.p = p

    def output_size(self, input_size=None):
        if type(input_size) == tuple:
            input_size = input_size[0]
        return input_size


class Pooling(PlaneLayer):

    def __init__(self, pool_size=1, stride=1, description='Pooling'):
        color = colors.green
        super(Pooling, self).__init__(color=color, description=description)
        self.pool_size = pool_size
        self.stride = stride

    def output_size(self, input_size=None):
        if type(input_size) == tuple:
            input_size = input_size[0]
        return input_size / self.pool_size



