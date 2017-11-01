Data:
matrix-model3.pickle and dan-model3.pickle are the best models in our experiments

data.train.txt, data.train.char are training data for word and character level models, respectively.

data.test.txt, data.test.char are testing data of Chinese sentence similarity

vector.sg300.small.ch are word embeddings with only words in training, testing and validation dataset

Data construction:
Training Dataset
To build the training dataset, we extracted the Chinese paraphrase sentences in machine translation evaluation corpora. Specifically, we used Chinese paraphrases in NIST2003 [https://catalog.ldc.upenn.edu/LDC2006T04] which contains 1100 English sentences with 4 Chinese translations, and CWMT2015 which contains 1859 English sentences with 4 Chinese translations. Moreover, we select phrase pairs in paraphrase sentences to extend the training corpus. Finally, we get 30846 paraphrases (nist-18187, cwmt-12659).

Testing and Development Dataset
To build the testing and development set in the Chinese sentence similarity task, we choose candidate sentences from the People’s Daily and Baidu encyclopedia corpora. To select sentence pairs with similarities ranging from high to low, we pair sentences in one paragraph and used averaged word embeddings as the sentence representation, to select high similarity sentence pairs. Then we delete unrelated sentences manually and randomly pair the left sentences to construct sentence pairs with low similarity. Finally, we got 1360 sentence pairs (1025 sentence pairs from the Baidu encyclopedia corpora and 335 sentence pairs from the People’s Daily). We collected human ratings of phrase similarity via an online questionnaire; participants were paid 7 cents for rating each sentence pair. In total, we obtained 104 valid questionnaires; every sentence pair was evaluated by 8 persons on average. The, we randomly partition the datasets into test and development set in 9:1.
