from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import ppdb_utils
import numpy as np
import utils

def sigmoid(x):
    return 1/(1+np.exp(-x))

def getSeqs(p1,p2,words):
    # p1 = p1.split()
    # p2 = p2.split()
    X1, X2 = [], []
    for i in p1:
        X1.append(ppdb_utils.lookupIDX(words,i))
    for i in p2:
        X2.append(ppdb_utils.lookupIDX(words,i))
    return X1, X2

def getSeqs2(p1,p2,words):
    # p1 = p1.split()
    # p2 = p2.split()
    X1 = []
    X2 = []
    for i in p1:
        tmp = []
	if len(i) == 0:
	    X1.append(tmp)
	else:
            for j in i:
                tmp.append(ppdb_utils.lookupIDX(words,j))
            X1.append(tmp)
    for i in p2:
        tmp = []
	if len(i) == 0:
	    X2.append(tmp)
	else:
            for j in i:
                tmp.append(ppdb_utils.lookupIDX(words, j))
            X2.append(tmp)
    return X1, X2

def getSeq(p1,words):
    p1 = p1.split()
    X1 = []
    for i in p1:
        X1.append(ppdb_utils.lookupIDX(words,i))
    return X1

def soft_weight_function(word_emb, char_emb):
    # #compute weights of char_emb with emb
    dots = (word_emb[:, :, None, :] * char_emb).sum(axis=3)  # 50*6*4
    norms_1 = np.sqrt(np.square(word_emb).sum(axis=2))  # 50*6
    norms_2 = np.sqrt(np.square(char_emb).sum(axis=3))  # 50*6*4
    weight = dots / (norms_1[:, :, None] * norms_2 + 1e-06)  # 50*6*4
    soft_weight = weight / (weight.sum(axis=2)[:, :, None] + 1e-06)
    return soft_weight

def weight_function(word_emb, char_emb):
    # #compute weights of char_emb with emb
    dots = (word_emb[:, :, None, :] * char_emb).sum(axis=3)  # 50*6*4
    norms_1 = np.sqrt(np.square(word_emb).sum(axis=2))  # 50*6
    norms_2 = np.sqrt(np.square(char_emb).sum(axis=3))  # 50*6*4
    weight = dots / (norms_1[:, :, None] * norms_2 + 1e-06)  # 50*6*4
    return weight

def getCorrelation(model,words,chars,f,params):
    f = open(f,'r')
    lines = f.readlines()
    preds = []
    golds = []
    seq1, char_seq1 = [], []
    seq2, char_seq2 = [], []
    for i in lines:
        i = i.split("\t")
        pp1 = i[0]; pp2 = i[1]; score = float(i[2])
	p1 = []; p2 = []
	char_p1 = []; char_p2 = []
	for j in pp1.split():
	    if len(j.split('_')) > 1:
		p1.append(j.split('_')[0])
		char_p1.append(j.split('_')[1:])
	    else:
		p1.append(j)
                char_p1.append([])
	for j in pp2.split():
            if len(j.split('_')) > 1:
                p2.append(j.split('_')[0])
                char_p2.append(j.split('_')[1:])
            else:
                p2.append(j)
                char_p2.append([])

        X1, X2 = getSeqs(p1,p2,words)
        char_X1, char_X2 = getSeqs2(char_p1, char_p2, chars)
        seq1.append(X1)
        seq2.append(X2)
        char_seq1.append(char_X1)
        char_seq2.append(char_X2)
        golds.append(score)
    x1,char_x1,m1,char_m1 = utils.prepare_data(seq1,char_seq1)
    x2,char_x2,m2,char_m2 = utils.prepare_data(seq2,char_seq2)

    scores = model.scoring_function(x1, char_x1, x2, char_x2, m1, char_m1, m2, char_m2)
    preds = np.squeeze(scores)
    return pearsonr(preds,golds)[0], spearmanr(preds,golds)[0]

def getAcc(model,words,f):
    f = open(f,'r')
    lines = f.readlines()
    preds = []
    golds = []
    seq1 = []
    seq2 = []
    ct = 0
    for i in lines:
        i = i.split("\t")
        p1 = i[0]; p2 = i[1]; score = i[2]
        X1, X2 = getSeqs(p1,p2,words)
        seq1.append(X1)
        seq2.append(X2)
        ct += 1
        if ct % 100 == 0:
            x1,m1 = utils.prepare_data(seq1)
            x2,m2 = utils.prepare_data(seq2)
            scores = model.scoring_function(x1,x2,m1,m2)
            scores = np.squeeze(scores)
            preds.extend(scores.tolist())
            seq1 = []
            seq2 = []
        golds.append(score)
    if len(seq1) > 0:
        x1,m1 = utils.prepare_data(seq1)
        x2,m2 = utils.prepare_data(seq2)
        scores = model.scoring_function(x1,x2,m1,m2)
        scores = np.squeeze(scores)
        preds.extend(scores.tolist())
    return acc(preds,golds)

def evaluate(model,words,file,params):
    p,s = getCorrelation(model,words,file)
    return p,s

def evaluate_all(model,words,chars,params):
    prefix = "../data/"
    parr = []; sarr = []

    farr = ["data.valid.char",
	    "data.test.char"]

    for i in farr:
        p,s = getCorrelation(model,words,chars,prefix+i,params)
        parr.append(p); sarr.append(s)

    s = ""
    for i,j,k in zip(parr,sarr,farr):
        s += str(i)+" "+str(j)+" "+k+" | "

    print s
    pp,ss = getCorrelation(model,words,chars,'../data/data.valid.char',params)
    return pp
