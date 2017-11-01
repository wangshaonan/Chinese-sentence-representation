import lasagne
from theano import tensor as T

class lasagne_matrix_layer(lasagne.layers.Layer):
    
    def __init__(self, incoming, num_units, nonlinearity, W, **kwargs):
        super(lasagne_matrix_layer, self).__init__(incoming, **kwargs)
	num_inputs = self.input_shape[2]
	self.num_units = num_units
	self.W = self.add_param(W, (num_inputs, num_units), name='W')
	self.nonlinearity = nonlinearity

    def get_output_for(self, input, **kwargs):
        return self.nonlinearity(T.dot(input, self.W))
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0],input_shape[1],self.num_units)
