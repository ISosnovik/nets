import numpy as np


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