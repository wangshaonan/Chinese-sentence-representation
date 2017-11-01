import ppdb_utils
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class tree(object):

    def __init__(self, phrase, words, chars):
        self.phrase = phrase
        self.embeddings = []
        self.char_embeddings = []
        self.representation = None
        self.char_representation = None

    def populate_embeddings(self, words, chars):
        phrase = self.phrase.lower()
	#print phrase
        arr = phrase.split()
	word = []
	char = []
	for i in arr:
	    if len(i.split('_')) > 1:
		word.append(i.split('_')[0])
		char.append(i.split('_')[1:])
	    else:
		word.append(i)
		char.append([])
        for i in word:
            self.embeddings.append(ppdb_utils.lookupIDX(words,i))
        for i in char:
	    tmp = []
            if len(i) == 0:
		self.char_embeddings.append([])
	    else:
                for j in i:
                    tmp.append(ppdb_utils.lookupIDX(chars, j))
                self.char_embeddings.append(tmp)

    def unpopulate_embeddings(self):
        self.embeddings = []
        self.char_embeddings = []
