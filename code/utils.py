import theano
import numpy as np
from theano import config
from time import time
from evaluate import evaluate
import cPickle
import sys

def checkIfQuarter(idx, n):
    if idx == round(n / 4.) or idx == round(n / 2.) or idx == round(3 * n / 4.):
        return True
    return False

def saveParams(model, fname):
    f = file(fname, 'wb')
    cPickle.dump(model.all_params, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def prepare_data(list_of_seqs, list_of_chars):
    lengths = [len(s) for s in list_of_seqs]
    chars_length = []
    for i in list_of_chars:
        tmp  = []
        for j in i:
            tmp.append(len(j))
        chars_length.append(tmp)
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths)
    #chars_maxlen = np.max(np.max(chars_length))
    chars_maxlen = 0
    for i in chars_length:
        if chars_maxlen < max(i):
            chars_maxlen = max(i)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    chars_x = np.zeros((n_samples, maxlen, chars_maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
    chars_x_mask = np.zeros((n_samples, maxlen, chars_maxlen)).astype(theano.config.floatX)
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype=config.floatX)

    for idx, s in enumerate(list_of_chars):
        for idxx, ss in enumerate(s):
            if not len(ss) == 0:
                #print 'prepare data: no chars'
                #continue
                chars_x[idx, idxx, :chars_length[idx][idxx]] = ss
                chars_x_mask[idx, idxx, :chars_length[idx][idxx]] = 1.
    chars_x_mask = np.asarray(chars_x_mask, dtype=config.floatX)
    return x, chars_x, x_mask, chars_x_mask


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def getDataSim(batch, nout):
    g1 = [];
    g2 = []
    for i in batch:
        g1.append(i[0].embeddings)
        g2.append(i[1].embeddings)

    g1x, g1mask = prepare_data(g1)
    g2x, g2mask = prepare_data(g2)

    scores = []
    for i in batch:
        temp = np.zeros(nout)
        score = float(i[2])
        ceil, fl = int(np.ceil(score)), int(np.floor(score))
        if ceil == fl:
            temp[fl - 1] = 1
        else:
            temp[fl - 1] = ceil - score
            temp[ceil - 1] = score - fl
        scores.append(temp)
    scores = np.matrix(scores) + 0.000001
    scores = np.asarray(scores, dtype=config.floatX)
    return (scores, g1x, g1mask, g2x, g2mask)


def train(model, train_data, dev, test, train, words, params):
        start_time = time()

        try:
            for eidx in xrange(params.epochs):

                kf = get_minibatches_idx(len(train_data), params.batchsize, shuffle=True)
                uidx = 0
                for _, train_index in kf:

                    uidx += 1
                    batch = [train_data[t] for t in train_index]

                    for i in batch:
                        i[0].populate_embeddings(words)
                        i[1].populate_embeddings(words)

                    (scores,g1x,g1mask,g2x,g2mask) = getDataSim(batch, model.nout)

                    cost = model.train_function(scores, g1x, g2x, g1mask, g2mask)

                    if np.isnan(cost) or np.isinf(cost):
                        print 'NaN detected'

                    #print 'Epoch ', (eidx+1), 'Update ', (uidx+1), 'Cost ', cost

                    #undo batch to save RAM
                    for i in batch:
                        i[0].representation = None
                        i[1].representation = None
                        i[0].unpopulate_embeddings()

                dp,ds = evaluate(model,words,dev,params)
                tp,ts = evaluate(model,words,test,params)
                rp,rs = evaluate(model,words,train,params)
                print "evaluation: ",dp,ds,tp,ts,rp,rs

                print 'Epoch ', (eidx+1), 'Cost ', cost
                sys.stdout.flush()

        except KeyboardInterrupt:
            print "Training interupted"

        end_time = time()
        print "total time:", (end_time - start_time)
