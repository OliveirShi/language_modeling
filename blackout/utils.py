import csv
import itertools
import numpy as np
import nltk
import sys
from os.path import isfile
import cPickle as pickle
import operator
import theano


def Q_dis(word2index,vocab,alpha):
    """
    create Q distribution for negative word sampling
    when alpha is 0:
    wc**alpha / sum(wc**alpha) -> uniform distrbution
    wc**1 / sum(wc**1) -> unigram distribution

    notice the q_dis do not include the <MASK/> and UNK
    """
    q_dis = []
    for item in vocab:
        print item[0],item[1]
        tmp = [word2index[item[0]]]*int(item[1]**alpha)
        # e.g. 'this' 342 -> [ 'this' , 'this' ,..,'this']
        q_dis.extend(tmp)

    return np.asarray(q_dis)

def Q_w(word2index,vocab,alpha):
    """
    weight for blackout the 1/relative frequence of the word
    """
    q_w = np.ones(len(vocab))

    q_t = 0
    for item in vocab:
        q_t = q_t + float(item[1]**alpha)

    for item in vocab:
        q_w[word2index[item[0]]] = float(item[1]**alpha)/float(q_t)

    return np.asarray(q_w,dtype=theano.config.floatX)

def blackout(q_dis,k,i):
    """
    sampling K negative word from q_dis, Sk != i
    """
    ne_sample = []

    while len(ne_sample) < k:
        p = np.random.randint(low=0,high=len(q_dis)-1,size=1)[0]
        if q_dis[p] == i:
            pass
        else:
            ne_sample.append(p)

    return np.asarray(ne_sample)


def negative_sample(y_train_i,k,q_dis):
    """
    negative sampling for integer vector y_train_i
    """
    neg_m = []
    for i in y_train_i:
        neg_m.append(blackout(q_dis,k,i))

    return np.asarray(neg_m)



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

class TextIterator:
    def __init__(self,source,maxlen,n_words_source=-1):

        self.source=open(source,'r')
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
        try:
            while True:
                s=self.source.readline()
                if s=="":
                    self.end_of_data=False
                    self.reset()
                    raise StopIteration
                s=s.strip().split(' ')

                if self.n_words_source>0:
                    s=[int(w) if int(w) <self.n_words_source else 3 for w in s]
                # filter long sentences
                if len(s)>self.maxlen:
                    continue
                return s

        except IOError:
            self.end_of_data=True

