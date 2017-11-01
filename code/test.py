import cPickle as pickle
import numpy as np
import sys

model = pickle.load(open(sys.argv[1]))

print model[0].get_value().shape
