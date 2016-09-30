import time
import sys
import os

import numpy as np
import theano
import theano.tensor as T

from rnnlm import *
from utils import *

lr=0.01
p=0.5
n_batch=10
NEPOCH=100

n_input=300
n_hidden=500
n_output=80000
cell='gru'
optimizer='sgd'
train_datafile='data/reddit-comments-2015.csv'
test_datafile='data/reddit-comments-2015.csv'
vocabulary_size=1000

disp_freq=100
sample_freq=200
save_freq=300


def train():
    # Load data
    train_data=TextIterator(train_datafile,vocab_file,n_word_source,n_batch=n_batch,maxlen=maxlen)
    test_data=TextIterator(test_datafile,vocab_file,n_words_source,n_batch=n_batch,maxlen=maxlen)

    model=RNNLM(n_input,n_hidden,n_output,cell,optimizer,p)
    print 'training...'
    start=time.time()
    g_error=999.99
    for epoch in xrange(NEPOCH):
        error=0
        idx=0
        in_start=time.time()
        for x,x_mask,y,y_mask in train_data:
            idx+=1
            cost=model.train(x,x_mask,y,y_mask,lr)
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


