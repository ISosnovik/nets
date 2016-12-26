from PIL import Image, ImageDraw
import numpy as np
import layers


class NN(object):

    def __init__(self):
        self.layers = []
        self.arrays = []
        self.images = []

    def add(self, layer):
        self.layers.append(layer)

    def _generate_arrays(self):
        factor = 1.0
        for layer in self.layers:
            arr = layer.draw(downsample_factor=factor)
            self.arrays.append(arr)
            if layer.__class__ is layers.Pooling:
                factor /= np.sqrt(layer.pool_size)
            if layer.__class__ is layers.Upsampling:
                factor *= np.sqrt(layer.factor)
            


    def draw(self):
        self._generate_arrays()

        images = [Image.fromarray(np.uint8(arr)) for arr in self.arrays]
        widths = [arr.shape[1] for arr in self.arrays]
        heights = [arr.shape[0] for arr in self.arrays]
        pic_height = max(heights)
        pic_width = sum(widths[1:]) * 0.5 + widths[0] 
        size = (int(pic_width), int(pic_height))

        nn_image = Image.new('RGBA', size, (255, 255, 255, 255))
        offset = (pic_width - self.arrays[-1].shape[1] * 2) / (len(self.arrays) - 1)
        x_padding = self.arrays[-1].shape[1] // 2
        offset = int(offset)
        for i, image in enumerate(images):
            y = (pic_height - self.arrays[i].shape[0]) / 2.0
            y = int(y)
            nn_image.paste(image, (x_padding + offset * i, y), image)
        nn_image.show()

