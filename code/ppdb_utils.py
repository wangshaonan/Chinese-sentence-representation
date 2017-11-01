import numpy as np
from tree import tree
import time
from random import randint
from random import choice
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import utils
import sys
from evaluate import evaluate_all


def lookupIDX(words,w):
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#","")
    if w in words:
        return words[w]
    else:
        return words['UUUNKKK']

def getPPDBData(f,words,chars):
    data = open(f,'r')
    lines = data.readlines()
    examples = []
    max_sent_len = 0
    max_char_len = 0
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split('\t')
            if len(i) == 3:
                e = (tree(i[0], words, chars), tree(i[1], words, chars))
                examples.append(e)
            else:
                print i
    print len(examples)
    return examples

def getWordmap(textfile):
    words={}
    We = []
    f = open(textfile,'r')
    lines = f.readlines()
    words['_null_'] = 0
    We.append(np.zeros(300))  # embedding size 300
    for (n,i) in enumerate(lines):
        n += 1
        i=i.split()
        j = 1
        v = []
        while j < len(i):
            v.append(float(i[j]))
            j += 1
        words[i[0]]=n
        We.append(v)
    return (words, np.array(We))

def getPairRand(d,idx):
    wpick = None
    ww = None
    while(wpick == None or (idx == ww)):
        ww = choice(d)
        ridx = randint(0,1)
        wpick = ww[ridx]
    return wpick

def getPairMixScore(d,idx,maxpair):
    r1 = randint(0,1)
    if r1 == 1:
        return maxpair
    else:
        return getPairRand(d,idx)

def getPairsFast(d, type):
    X = []
    T = []
    pairs = []
    for i in range(len(d)):
        (p1,p2) = d[i]
        X.append(p1.representation)
        X.append(p2.representation)
        T.append(p1)
        T.append(p2)

    arr = pdist(X,'cosine')
    arr = squareform(arr)

    for i in range(len(arr)):
        arr[i,i]=1
        if i % 2 == 0:
            arr[i,i+1] = 1
        else:
            arr[i,i-1] = 1

    arr = np.argmin(arr,axis=1)
    for i in range(len(d)):
        (t1,t2) = d[i]
        p1 = None
        p2 = None
        if type == "MAX":
            p1 = T[arr[2*i]]
            p2 = T[arr[2*i+1]]
        if type == "RAND":
            p1 = getPairRand(d,i)
            p2 = getPairRand(d,i)
        if type == "MIX":
            p1 = getPairMixScore(d,i,T[arr[2*i]])
            p2 = getPairMixScore(d,i,T[arr[2*i+1]])
        pairs.append((p1,p2))
    return pairs

def getpairs(model, batch, params):
    g1, char_g1 = [], []
    g2, char_g2 = [], []

    for i in batch:
        g1.append(i[0].embeddings)
        g2.append(i[1].embeddings)
        char_g1.append(i[0].char_embeddings)
        char_g2.append(i[1].char_embeddings)

    g1x, char_g1x, g1mask, char_g1mask = utils.prepare_data(g1, char_g1)
    g2x, char_g2x, g2mask, char_g2mask = utils.prepare_data(g2, char_g2)
    char_embg1 = model.char_embg_function(char_g1x)  # 50*6*4*300
    word_embg1 = model.word_embg_function(g1x)  # 50*6*300
    char_embg2 = model.char_embg_function(char_g2x)  # 50*7*4*300
    word_embg2 = model.word_embg_function(g2x)  # 50*7*300


    embg1 = model.feedforward_function(g1x, char_g1x, g1mask, char_g1mask)  #50*40,  50*40*6,  50*40, 50*40*6 -- 50*300
    embg2 = model.feedforward_function(g2x, char_g2x, g2mask, char_g2mask)

    for idx, i in enumerate(batch):
        i[0].representation = embg1[idx, :]
        i[1].representation = embg2[idx, :]

    pairs = getPairsFast(batch, params.type)
    p1, char_p1 = [],[]
    p2, char_p2 = [],[]
    for i in pairs:
        p1.append(i[0].embeddings)
        p2.append(i[1].embeddings)
        char_p1.append(i[0].char_embeddings)
        char_p2.append(i[1].char_embeddings)

    p1x, char_p1x, p1xmask, char_p1xmask = utils.prepare_data(p1, char_p1)
    p2x, char_p2x, p2xmask, char_p2xmask = utils.prepare_data(p2, char_p2)

    return (g1x, char_g1x, g1mask, char_g1mask, g2x, char_g2x, g2mask, char_g2mask, p1x, char_p1x,
            p1xmask, char_p1xmask, p2x, char_p2x, p2xmask, char_p2xmask)


def train(model, data, words, chars, params):
    start_time = time.time()

    counter = 0
    tmp = 0
    try:
        for eidx in xrange(params.epochs):

            kf = utils.get_minibatches_idx(len(data), params.batchsize, shuffle=True)
            uidx = 0
            for _, train_index in kf:

                uidx += 1

                batch = [data[t] for t in train_index]
                for i in batch:
                    i[0].populate_embeddings(words, chars)
                    i[1].populate_embeddings(words, chars)

                (g1x, char_g1x, g1mask, char_g1mask, g2x, char_g2x, g2mask, char_g2mask,
                 p1x, char_p1x, p1mask, char_p1mask, p2x, char_p2x, p2mask, char_p2mask) = getpairs(model, batch, params)

                char_embg1 = model.char_embg_function(char_g1x)  # 50*6*4*300
                word_embg1 = model.word_embg_function(g1x)  # 50*6*300
                char_embg2 = model.char_embg_function(char_g2x)  # 50*6*4*300
                word_embg2 = model.word_embg_function(g2x)  # 50*6*300
                char_embp1 = model.char_embg_function(char_p1x)  # 50*6*4*300
                word_embp1 = model.word_embg_function(p1x)  # 50*6*300
                char_embp2 = model.char_embg_function(char_p2x)  # 50*6*4*300
                word_embp2 = model.word_embg_function(p2x)  # 50*6*300

		cost = model.train_function(g1x, char_g1x, g2x, char_g2x, p1x, char_p1x, p2x, char_p2x, g1mask, char_g1mask, g2mask, char_g2mask, p1mask, char_p1mask, p2mask, char_p2mask)

                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected'

                if (utils.checkIfQuarter(uidx, len(kf))):
                    if (params.save):
                        counter += 1
                        utils.saveParams(model, params.outfile + str(counter) + '.pickle')
                    if (params.evaluate):
                        pp = evaluate_all(model, words, chars,params)
			if tmp < pp:
                            tmp = pp
                            utils.saveParams(model, params.outfile+'.pickle')
                        sys.stdout.flush()

                #undo batch to save RAM
                for i in batch:
                    i[0].representation = None
                    i[1].representation = None
                    i[0].unpopulate_embeddings()
                    i[1].unpopulate_embeddings()

                #print 'Epoch ', (eidx+1), 'Update ', (uidx+1), 'Cost ', cost

            if (params.save):
                counter += 1
                utils.saveParams(model, params.outfile + str(counter) + '.pickle')

            if (params.evaluate):
                pp = evaluate_all(model, words, chars, params)
		if tmp < pp:
                    tmp = pp
                    utils.saveParams(model, params.outfile+'.pickle')

            print 'Epoch ', (eidx + 1), 'Cost ', cost

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.time()
    print "total time:", (end_time - start_time)
