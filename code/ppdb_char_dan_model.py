import theano
import numpy as np
from theano import tensor as T
from theano import config
import lasagne
from lasagne_average_layer import lasagne_average_layer

from word_representation_layer1 import word_representation_layer1
from word_representation_layer2 import word_representation_layer2
from word_representation_layer3 import word_representation_layer3

class ppdb_char_dan_model(object):

    def __init__(self, We_initial, Wc_initial, params):

        initial_We = theano.shared(np.asarray(We_initial, dtype = config.floatX))
        We = theano.shared(np.asarray(We_initial, dtype = config.floatX))
        initial_Wc = theano.shared(np.asarray(Wc_initial, dtype = config.floatX))
        Wc = theano.shared(np.asarray(Wc_initial, dtype = config.floatX))

        g1batchindices = T.imatrix(); char_g1batchindices = T.itensor3(); g2batchindices = T.imatrix(); char_g2batchindices = T.itensor3()
        p1batchindices = T.imatrix(); char_p1batchindices = T.itensor3(); p2batchindices = T.imatrix(); char_p2batchindices = T.itensor3()
        g1mask = T.matrix(); char_g1mask = T.tensor3(); g2mask = T.matrix(); char_g2mask = T.tensor3()
        p1mask = T.matrix(); char_p1mask = T.tensor3(); p2mask = T.matrix(); char_p2mask = T.tensor3()

        l_in = lasagne.layers.InputLayer((None, None))
        l_char_in = lasagne.layers.InputLayer((None, None, None))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        l_char_mask = lasagne.layers.InputLayer(shape=(None, None, None))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=We.get_value().shape[0], output_size=We.get_value().shape[1], W=We)
        l_char_emb = lasagne.layers.EmbeddingLayer(l_char_in, input_size=Wc.get_value().shape[0],
                                                   output_size=Wc.get_value().shape[1], W=Wc)    #50*6*4*300

	char_embg11 = lasagne.layers.get_output(l_char_emb, {l_char_in: char_g1batchindices})
	word_embg11 = lasagne.layers.get_output(l_emb, {l_in:g1batchindices})
	self.char_embg_function = theano.function([char_g1batchindices], char_embg11)
	self.word_embg_function = theano.function([g1batchindices], word_embg11)
        #char_embg1 = lasagne.layers.get_output(l_char_emb, {l_char_in: char_g1batchindices})
        #self.char_representation_function = theano.function([char_g1batchindices], char_embg1)
	if params.nntype == 'dan_char1':
            l_word_representation = word_representation_layer1([l_emb, l_char_emb, l_char_mask]) #lasagne.nonlinearities.tanh
	elif params.nntype == 'dan_char2':
	    l_word_representation = word_representation_layer2([l_emb, l_char_emb, l_char_mask])
	elif params.nntype == 'dan_char3':
            l_word_representation = word_representation_layer3([l_emb, l_char_emb, l_char_mask])
	else:
	    print 'something wrong in ppdb_char_dan_model !'

	l_average = lasagne_average_layer([l_word_representation, l_mask])
        l_dan = lasagne.layers.DenseLayer(l_average, params.hiddensize, W=lasagne.init.Normal(0.1), b=lasagne.init.Constant(0.), nonlinearity=params.nonlinearity)

        embg1 = lasagne.layers.get_output(l_dan, {l_in:g1batchindices, l_mask:g1mask, l_char_in:char_g1batchindices, l_char_mask:char_g1mask})
        embg2 = lasagne.layers.get_output(l_dan, {l_in:g2batchindices, l_mask:g2mask, l_char_in:char_g2batchindices, l_char_mask:char_g2mask})
        embp1 = lasagne.layers.get_output(l_dan, {l_in:p1batchindices, l_mask:p1mask, l_char_in:char_p1batchindices, l_char_mask:char_p1mask})
        embp2 = lasagne.layers.get_output(l_dan, {l_in:p2batchindices, l_mask:p2mask, l_char_in:char_p2batchindices, l_char_mask:char_p2mask})

        g1g2 = (embg1*embg2).sum(axis=1)
        g1g2norm = T.sqrt(T.sum(embg1**2,axis=1)) * T.sqrt(T.sum(embg2**2,axis=1))
        g1g2 = g1g2 / g1g2norm

        p1g1 = (embp1*embg1).sum(axis=1)
        p1g1norm = T.sqrt(T.sum(embp1**2,axis=1)) * T.sqrt(T.sum(embg1**2,axis=1))
        p1g1 = p1g1 / p1g1norm

        p2g2 = (embp2*embg2).sum(axis=1)
        p2g2norm = T.sqrt(T.sum(embp2**2,axis=1)) * T.sqrt(T.sum(embg2**2,axis=1))
        p2g2 = p2g2 / p2g2norm

        costp1g1 = params.margin - g1g2 + p1g1
        costp1g1 = costp1g1*(costp1g1 > 0)

        costp2g2 = params.margin - g1g2 + p2g2
        costp2g2 = costp2g2*(costp2g2 > 0)

        cost = costp1g1 + costp2g2
        network_params = lasagne.layers.get_all_params(l_dan, trainable=True)
        network_params.pop(0)
        self.all_params = lasagne.layers.get_all_params(l_dan, trainable=True)

        #regularization
        l2 = 0.5*params.LC*sum(lasagne.regularization.l2(x) for x in network_params)
        if params.updatewords:
            word_reg = 0.5*params.LW*lasagne.regularization.l2(We-initial_We)
            char_reg = 0.5*params.LWC*lasagne.regularization.l2(Wc-initial_Wc)
            cost = T.mean(cost) + l2 + word_reg +char_reg
        else:
            cost = T.mean(cost) + l2

        self.feedforward_function = theano.function([g1batchindices, char_g1batchindices, g1mask, char_g1mask], embg1)
        self.cost_function = theano.function([g1batchindices, char_g1batchindices, g2batchindices, char_g2batchindices,
                                              p1batchindices, char_p1batchindices, p2batchindices, char_p2batchindices,
                                              g1mask, char_g1mask, g2mask, char_g2mask, p1mask, char_p1mask, p2mask, char_p2mask], cost)

        prediction = g1g2

        self.scoring_function = theano.function([g1batchindices, char_g1batchindices, g2batchindices, char_g2batchindices,
                             g1mask, char_g1mask, g2mask, char_g2mask],prediction)

        self.train_function = None
        if params.updatewords:
            grads = theano.gradient.grad(cost, self.all_params)
            if params.clip:
                grads = [lasagne.updates.norm_constraint(grad, params.clip, range(grad.ndim)) for grad in grads]
            updates = params.learner(grads, self.all_params, params.eta)
            self.train_function = theano.function([g1batchindices, char_g1batchindices, g2batchindices, char_g2batchindices,
                            p1batchindices, char_p1batchindices, p2batchindices, char_p2batchindices,
                            g1mask, char_g1mask, g2mask, char_g2mask, p1mask, char_p1mask, p2mask, char_p2mask], cost, updates=updates)
        else:
            self.all_params = network_params
            grads = theano.gradient.grad(cost, self.all_params)
            if params.clip:
                grads = [lasagne.updates.norm_constraint(grad, params.clip, range(grad.ndim)) for grad in grads]
            updates = params.learner(grads, self.all_params, params.eta)
            self.train_function = theano.function([g1batchindices, char_g1batchindices, g2batchindices, char_g2batchindices,
                            p1batchindices, char_p1batchindices, p2batchindices, char_p2batchindices,
                            g1mask, char_g1mask, g2mask, char_g2mask, p1mask, char_p1mask, p2mask, char_p2mask], cost, updates=updates)
