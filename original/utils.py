import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle
import os
import operator
from os.path import isfile,join
import csv
import itertools
import gzip
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


def fopen(filename,mode='r'):
    if filename.endwith('gz'):
        return gzip.open(filename,mode)
    return open(filename,mode)

class TextIterator:
    def __init__(self,source,source_dict,n_batch,maxlen,n_words_source=-1):

        with open(source,'rb')as f:
            self.source=pickle.load(file(source))
        with open(source_dict,'rb')as f:
            self.source_dict=pickle.load(f)

        self.n_batch=n_batch
        self.maxlen=maxlen
        self.n_words_source=n_words_source
        self.end_of_data=False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data=False
            self.reset()
            raise StopIteration
        source=[]
        try:
            while True:
                ss=self.source.readline()
                if ss=="":
                    raise IOError
                ss=ss.strip().split()
                ## filter oov words
                ss=[self.source_dict[w] if w in self.source_dict else 1
                    for w in ss]
                if self.n_words_source>0:
                    ss=[w if w <self.n_words_source else 1 for w in ss]
                ## filter long sentences
                if len(ss)>self.maxlen:
                    continue

                source.append(ss)
                if len(source)>=self.n_batch:
                    break
        except IOError:
            self.end_of_data=True

        if len(source)<=0:
            self.end_of_data=False
            self.reset()
            raise StopIteration
        return prepare_data(source)

def prepare_data(seqs_x):
    lengths_x=[len(s)-1 for s in seqs_x]
    n_samples=len(seqs_x)
    maxlen_x=np.max(lengths_x)+1

    x=np.zeros((maxlen_x,n_samples)).astype('int32')
    y=np.zeros((maxlen_x,n_samples)).astype('int32')
    x_mask=np.zeros((maxlen_x,n_samples)).astype('float32')
    y_mask=np.zeros((maxlen_x,n_samples)).astype('float32')

    for idx,s_x in enumerate(seqs_x):
        x[:lengths_x[idx],idx]=s_x[:-1]
        y[:lengths_x[idx],idx]=s_x[1:]
        x_mask[:lengths_x[idx],idx]=1
        y_mask[:lengths_x[idx],idx]=1

    return x,x_mask,y,y_mask





def gengerate_data(filename="data/reddit-comments-2015.csv", vocabulary_size=2000, min_sent_characters=0):

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

    sorted_vocab.append((UNKNOWN_TOKEN,1))

    index_to_word = [x[0] for x in sorted_vocab]
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

    # Create the training data
    x = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])

    with open('data/dataset.pkl','wb')as f:
        pickle.dump(x,f)
    with open('data/word2index.pkl','wb')as f:
        pickle.dump(word_to_index,f)
    with open('data/index2word.pkl','wb')as f:
        pickle.dump(index_to_word,f)
    with open('data/vocab.pkl','wb')as f:
        pickle.dump( sorted_vocab,f)



