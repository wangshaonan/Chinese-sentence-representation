import lasagne
from theano import tensor as T

class word_representation_layer2(lasagne.layers.MergeLayer):
    def __init__(self, incoming, **kwargs):
        super(word_representation_layer2, self).__init__(incoming, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        emb = inputs[0] #batch*sent*300        #50*6*300
        char_emb = inputs[1]  #batch*sent*num_char*300       #50*6*4*300
        char_mask = inputs[2]  #batch*sent*num_char        #50*6*4
	char_emb = (char_emb*char_mask[:,:,:,None]).sum(axis=2) / (char_mask.sum(axis=2)[:,:,None]+10e-8)
        emb = T.concatenate([emb[:,:,:,None], char_emb[:,:,:,None]], axis=3).max(axis=3)
	#emb = 0.5*(emb+char_emb)
        return emb

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2])
