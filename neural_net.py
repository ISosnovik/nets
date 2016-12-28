from PIL import Image, ImageDraw
import numpy as np
import layers


class NN(object):

    def __init__(self):
        self.layers = []
        self.images = []
        self.arrays = None
        self.offsets = None

    def add(self, layer):
        self.layers.append(layer)

    def _resize_layers(self):
        height = self.layers[0].height
        for layer in self.layers[1:]:
            layer.height = height
            if type(layer) is layers.Pooling:
                height /= np.sqrt(layer.pool_size)
            if type(layer) is layers.Upsampling:
                height *= np.sqrt(layer.factor)
            height = int(height)

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
        for h, layer in zip(heights, self.layers):
            if not h == current_height:
                current_offset += sparse_distance
            x_positions.append(current_offset)
            current_offset += layer.width * 2 + interlayer
            current_height = h

        pic_width = x_positions[-1] + widths[-1] + padding
        size = (int(pic_width), int(pic_height))
        return size, x_positions, y_positions
 
    def draw(self, interlayer=20, sparse_distance=20):
        self._resize_layers()
        size, x_, y_ = self._layout_layers(interlayer, sparse_distance)
        nn_image = Image.new('RGBA', size, (255, 255, 255, 255))
        for x, y, image in zip(x_, y_, self.images):
            nn_image.paste(image, (int(x), int(y)), image)
        nn_image.show()






