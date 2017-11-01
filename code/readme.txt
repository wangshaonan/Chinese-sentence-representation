This code provides an architecture for sentence representation models with word and character level information. The code is written in python and requires numpy, scipy, theano and the lasagne library. The code is modified from https://github.com/jwieting/iclr2016

The code consists of five different word embedding methods with five sentence composition models.

Word embedding method module:
word_representation_layer1.py add mask gate operation on characters
word_representation_layer2.py add max pooling operation on words
word_representation_layer3.py add mask gate and pooling operation on both characters and words
word_representation_layer4.py add mask gate operation on words
word_representation_layer5.py add mask gate operation on both characters and words

Sentence composition model module:
ppdb_char_word_model.py Average model
ppdb_char_dan_model.py Dan model
ppdb_char_matrix_model.py Matrix model
ppdb_char_rnn_model.py RNN model
ppdb_char_lstm_model.py LSTM mdoel

See run_train.sh for an example of usage.