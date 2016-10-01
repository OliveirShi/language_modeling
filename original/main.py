import time
import sys
import os
import cPickle as pickle
import numpy as np
import theano
import theano.tensor as T

from rnnlm import *
from utils import TextIterator,save_model

lr=0.01
p=0.5
n_batch=10
NEPOCH=100

n_input=300
n_hidden=500
maxlen=50
cell='gru'
optimizer='sgd'
train_datafile='data/dataset.pkl'
test_datafile='data/dataset.pkl'
vocab_file='data/vocab.pkl'
vocabulary_size=1000

disp_freq=100
sample_freq=200
save_freq=300


def train():
    print 'building...'
    model=RNNLM(n_input,n_hidden,vocabulary_size,cell,optimizer,p)

    # Load data
    print 'loading...'
    train_data=TextIterator(train_datafile,vocab_file,n_words_source=vocabulary_size,n_batch=n_batch,maxlen=maxlen)
    #test_data=TextIterator(test_datafile,vocab_file,n_words_source=vocabulary_size,n_batch=n_batch,maxlen=maxlen)


    print 'training...'
    start=time.time()
    for epoch in xrange(NEPOCH):
        error=0
        idx=0
        in_start=time.time()
        for x,x_mask,y,y_mask in train_data:
            idx+=1
            cost=model.train(x,x_mask,y,y_mask,n_batch,lr)
            error+=cost
            if np.isnan(cost) or np.isinf(cost):
                print 'Not a Number Or Infinity detected!'
            if idx % disp_freq==0:
                print 'epoch:',epoch,'idx:',idx,'cost:',error/disp_freq

            if idx%save_freq==0:
                print 'dumping...'
                with open('data/parameters_%.2f.pkl'%(time.time()-start),'wb')as f:
                    pickle.dump(model.params,f)
            if idx % sample_freq==0:
                print 'Sampling....'
                y_pred=model.predict(x,x_mask,y,y_mask)
                print y_pred

    print "Finished. Time = "+str(time.time()-start)

    print "save model..."
    save_model("./model/rnnlm.model",model)

if __name__ == '__main__':
    train()
