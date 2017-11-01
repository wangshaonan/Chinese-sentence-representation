import lasagne
from theano import tensor as T

class word_representation_layer5(lasagne.layers.MergeLayer):
    def __init__(self, incoming,num_units=300, W1=lasagne.init.Normal(), b1=lasagne.init.Normal(), W2=lasagne.init.Normal(), b2=lasagne.init.Normal(), **kwargs):
        super(word_representation_layer5, self).__init__(incoming, **kwargs)
        self.W1 = self.add_param(W1, (600,300))
        self.b1 = self.add_param(b1, (300,))
        self.W2 = self.add_param(W2, (600,300))
        self.b2 = self.add_param(b2, (300,))

    def get_output_for(self, inputs, **kwargs):
        emb = inputs[0] #batch*sent*300        #50*6*300
        char_emb = inputs[1]  #batch*sent*num_char*300       #50*6*4*300
        char_mask = inputs[2]  #batch*sent*num_char        #50*6*4
        #computing weights
        emb2 = emb.reshape((emb.shape[0], emb.shape[1], 1, emb.shape[2]))
        emb2 = emb2.repeat(char_emb.shape[2], axis=2)
        attention_weight = lasagne.nonlinearities.tanh(T.dot(T.concatenate([emb2, char_emb], axis=3), self.W1)+self.b1)
	attention_weight = attention_weight + 1
	#attention
        char_emb = (char_emb * char_mask[:, :, :, None]*attention_weight).sum(axis=2)
        #attention2
        attention_weight2 = lasagne.nonlinearities.sigmoid(T.dot(T.concatenate([emb, char_emb], axis=2), self.W2)+self.b2)
        emb = emb*attention_weight2 + char_emb*(1-attention_weight2)
        return emb

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2])
