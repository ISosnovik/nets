import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
    
from utils import *
import colors



# General Class
class DenseLayer(object):

    def __init__(self, num_units=20, color=colors.dark_grey, description='Dense', radius=10):
        self.color = color
        self.description = description
        self.num_units = num_units
        self.radius = radius
        self.num_render = 10

    @property
    def legend(self):
        patch = patches.Circle((0,0), 1., facecolor=self.color)
        return (patch, self.description)

    @property
    def width(self):
        step = self.radius * 0.5
        w = self.radius / 2.0 + (self.num_render - 1) * step
        return w * 1.0



    def draw(self, lw=2, figsize=(10, 10)): 
        tan = 2.4
        step = self.radius * 0.5
        x_pos = [self.radius / 2.0 + step * i for i in range(self.num_render)]
        y_pos = [self.radius / 2.0 + tan * step * i for i in range(self.num_render)]

        circles = [patches.Circle((x, y), self.radius, lw=lw, facecolor=self.color) for x, y in zip(x_pos, y_pos)]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # draw layer
        for circle in circles[::-1]:
            ax.add_patch(circle)

        ax.set_xlim(- 10, x_pos[-1] + self.radius / 2.0 + 10)
        ax.set_ylim(- 20, 260)
        plt.axes().set_aspect('equal')
        plt.axis('off')
        fig.canvas.draw()
        
        data = figure2array(fig)
        plt.close(fig)
        return data



