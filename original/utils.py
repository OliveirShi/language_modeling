import numpy as np
import cPickle as pickle
import operator
import csv
import itertools
import gzip
import nltk

SENTENCE_START_TOKEN='<s>'
SENTENCE_END_TOKEN='</s>'
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
    if filename.endswith('gz'):
        return gzip.open(filename,mode)
    return open(filename,mode)

class TextIterator:
    def __init__(self,source,index2word_file,n_batch,maxlen,n_words_source=-1):

        self.source=fopen(source)
        with open(index2word_file,'rb')as f:
            self.index2word=pickle.load(f)

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
                ss=ss.strip().split(' ')

                if self.n_words_source>0:
                    ss=[int(w) if int(w) <self.n_words_source else 3 for w in ss]
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
    maxlen_x=np.max(lengths_x)

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


