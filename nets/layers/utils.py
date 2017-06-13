

def singular(x):
    return len(set(x)) <= 1


def merge_input_shapes(input_shapes, merge):
    if len(input_shapes) == 1:
        return input_shapes[0]
        
    ndims = [len(shape) for shape in input_shapes]
    if not singular(ndims):
        raise ValueError('All inputs must have '
                         'the same dimensionality')

    if merge == 'add':
        if not singular(input_shapes):
            raise ValueError('All inputs must have the '
                             'same shapes for "add" mode')
        return input_shapes[0]
    
    if merge == 'concat':
        sizes = [s[:-1] for s in input_shapes]
        if not singular(sizes):
            raise ValueError('All inputs must have the same '
                             'shapes but the number of '
                             'channels for "concat" mode')
        n_channels = sum((shape[-1] for shape in input_shapes))
        return sizes + (n_channels, )

