from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict



from PlaneLayer import *
from DenseLayer import *


class NN(object):

    def __init__(self):
        self.layers = []
        self.images = []
        self.arrays = None
        self.offsets = None
        self.nn_image = None
        self.legend = ()
        self.legend_kwargs = {}

    def add(self, layer):
        self.layers.append(layer)

    def _resize_plane_layers(self):
        plane_layers = [l for l in self.layers if not type(l) == DenseLayer]

        input_size = plane_layers[0].input_size
        try:
            input_size = input_size[0]
        except TypeError:
            pass
        inputs = [input_size]
        outputs = []
        for layer in plane_layers:
            output_size = layer.output_size(input_size)
            outputs.append(output_size)
            input_size = output_size
            inputs.append(input_size)
        
        max_in = max(inputs)
        max_out = max(outputs)
        for layer, in_, out_ in zip(plane_layers, inputs, outputs):
            if type(layer) == Upsampling:
                layer.height = 150 * out_ / max_out
            else:
                layer.height = 150 * in_ / max_in



    def _resize_dense_layers(self):
        dense_layers = [l for l in self.layers if type(l) == DenseLayer]
        for l in dense_layers:
            l.num_render = l.num_units


    def _layout_layers(self, interlayer, sparse_distance):
        self.arrays = [layer.draw() for layer in self.layers]
        self.images = [Image.fromarray(arr) for arr in self.arrays]
        widths = [arr.shape[1] for arr in self.arrays]
        heights = [arr.shape[0] for arr in self.arrays]
        pic_height = max(heights)
        y_positions = [(pic_height - h) / 2.0 for h in heights]
        x_positions = []
        padding = 100
        current_offset = padding
        current_height = heights[0]
        for i, h in enumerate(heights):
            layer = self.layers[i]
            if not type(layer) == DenseLayer:
                if not h == current_height:
                        current_offset += sparse_distance
                x_positions.append(current_offset)
                current_offset += layer.width * 2 + interlayer
                current_height = h
            else:
                x_positions.append(current_offset)
                try:
                    current_h = layer.num_render
                    next_h = self.layers[i+1].num_render
                    if next_h < current_h:
                        delta = (current_h - next_h) / 2.0 
                        delta *= layer.radius /  2.6
                        delta += 80.0
                        current_offset += delta
                    else:
                        delta = (current_h - next_h) / 2.0 
                        delta *= layer.radius /  2.6
                        delta += 40.0
                        current_offset += delta 
                except AttributeError:
                    current_h = layer.num_render
                    off = layer.num_render * layer.radius / 2 / 2.6
                    current_offset += 4 * off + interlayer


        pic_width = x_positions[-1] + widths[-1] + padding
        size = (int(pic_width), int(pic_height))
        return size, x_positions, y_positions
 
    def compile(self, interlayer=20, sparse_distance=20):
        self._resize_plane_layers()
        self._resize_dense_layers()
        size, x_, y_ = self._layout_layers(interlayer, sparse_distance)
        self.nn_image = Image.new('RGBA', size, (255, 255, 255, 255))
        for x, y, image in zip(x_, y_, self.images):
            self.nn_image.paste(image, (int(x), int(y)), image)
        
    def add_legend(self, ncol=10, mode="expand", borderaxespad=0., 
        loc=8, bbox_to_anchor=(0.1, -0.1, 0.8, .102), fontsize=14):

        legends = {l.legend[1]: l.legend[0] for l in self.layers}
        colors = legends.values()
        labels = legends.keys()
        self.legend = (colors, labels)
        self.legend_kwargs = {
            'ncol': ncol, 
            'mode': mode, 
            'borderaxespad': borderaxespad, 
            'loc': loc, 
            'bbox_to_anchor': bbox_to_anchor, 
            'fontsize': fontsize,
            'handler_map': {
                patches.Rectangle: HandlerRect(),
                patches.Circle: HandlerCircle()    
            }
        }

    def draw(self, figsize=(20,40), save=False, save_path=None):
        image = self.nn_image.convert('RGB')
        image = np.array(image)

        fig = plt.figure(figsize=figsize)
        plt.imshow(image)
        plt.legend(*self.legend, **self.legend_kwargs)
        plt.axis('off')
        if save:
            plt.savefig(save_path)
        plt.show()







