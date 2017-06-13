from __future__ import print_function, division
from collections import Iterable


class Layer:
    '''Basic class'''

    def __init__(self, incoming=None, input_shape=None, style={}, description=None):
        '''
            incoming: incoming Layer or list/tuple of Layers
            style: dictionary with the defined style attributes: shape, color, etc. 
            input_shape: tuple. shape of the input data. 
                Could be None, if the incoming layer is provided
            description: the name of the layer. 
        '''
        self.style = style
        self.description = description

        if incoming:
            if not isinstance(incoming, Iterable):
                incoming = [incoming]
            self.incoming = list(incoming)
            self.input_shape = None
        else if input_shape:
            self.incoming = None
            self.input_shape = input_shape
        else:
            raise ValueError("Neither incoming nor input_shape are provided")
               
    
    def get_output_shape(self):
        raise NotImplementedError
            


