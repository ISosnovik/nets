import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerPatch


class HandlerRect(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        xy = (width - height) / 2.0, 0
        width = height
        height = height
        p = patches.Rectangle(xy, width, height)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


class HandlerCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        xy = width / 2.0, height / 2.0
        r = height / 1.5
        p = patches.Circle(xy, r)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]