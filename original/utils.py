import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle
import os
import operator
from os.path import isfile,join
import csv
import itertools
import nltk

SENTENCE_START_TOKEN='<S>'
SENTENCE_END_TOKEN='</S>'
UNKNOWN_TOKEN='<unk>'

def save_model(f,model):
    ps={}
    for p in model.params:
        ps[p.name]=p.get_value()
    pickle.dump(ps,open(f,'wb'))

def load_model(f,model):
    ps=pickle.load(open(f,'rb'))
    for p in model.params:
        p.set_value(ps[p.name])
    return model


def load_data(filename="data/reddit-comments-2015-08.csv", vocabulary_size=2000, min_sent_characters=0):
    if isfile('data/dataset.pkl'):
        with open('data/dataset.pkl')as f:
            (x, maskx,y,masky, word_to_index, index_to_word, sorted_vocab)=pickle.load(f)
            return x, maskx,y,masky, word_to_index, index_to_word, sorted_vocab

    word_to_index = []
    index_to_word = []

    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print("Reading CSV file...")

    with open(filename, 'rt') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode("utf-8").lower()) for x in reader])
        # Filter sentences
        sentences = [s for s in sentences if len(s) >= min_sent_characters]
        sentences = [s for s in sentences if "http" not in s]
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (SENTENCE_START_TOKEN, x, SENTENCE_END_TOKEN) for x in sentences]
    print("Parsed %d sentences." % (len(sentences)))


    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)[:vocabulary_size-2]

    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    sorted_vocab = sorted(vocab, key=operator.itemgetter(1))

    # concern about blackout algorithm
    sorted_vocab.append(("<MASK/>",1))
    sorted_vocab.append((UNKNOWN_TOKEN,1))

    index_to_word = [x[0] for x in sorted_vocab]
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

    # Create the training data
    x = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    maskx = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    masky= np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    with open('data/dataset.pkl')as f:
        pickle.dump((x,maskx,y,masky, word_to_index, index_to_word, sorted_vocab),f)

    return x, maskx,y,masky, word_to_index, index_to_word, sorted_vocab

