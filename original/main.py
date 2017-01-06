import time

from rnnlm import *
from utils import TextIterator,save_model

import logging


lr=0.5
p=0
n_batch=100
NEPOCH=100

n_input=100
n_hidden=250
maxlen=100
cell='gru'
optimizer='sgd'
train_datafile='../ptb/idx_ptb.train.txt'
valid_datafile='../ptb/idx_ptb.valid.txt'
test_datafile='../ptb/idx_ptb.test.txt'
n_words_source=-1
vocabulary_size=10001

disp_freq=100
sample_freq=200
save_freq=5000
clip_freq=2000

def train(lr):
    # Load data
    print 'loading dataset...'
    train_data=TextIterator(train_datafile,n_words_source=n_words_source,n_batch=n_batch)
    test_data=TextIterator(test_datafile,n_words_source=n_words_source,n_batch=n_batch)

    print 'building model...'
    model=RNNLM(n_input,n_hidden,vocabulary_size,cell,optimizer,p)
    print 'training start...'
    start=time.time()
    for epoch in xrange(NEPOCH):
        error=0
        idx=0
        in_start=time.time()
        for x,x_mask,y,y_mask in train_data:

            idx+=1
            beg_time=time.time()
            print x.shape
            cost=model.train(x,x_mask,y,y_mask,n_batch,lr)
            print 'index:',idx,'time:',time.time()-beg_time,'cost:',cost,'lr:',lr
            error+=np.sum(cost)
            if np.isnan(cost) or np.isinf(cost):
                print 'NaN Or Inf detected!'
                return -1
            if idx % disp_freq==0:
                print 'epoch:',epoch,'idx:',idx,'cost:',error/disp_freq
                error=0
            if idx%save_freq==0:
                print 'dumping...'
                save_model('./model/parameters_%.2f.pkl'%(time.time()-start),model)
            if idx % sample_freq==0:
                print 'Sampling....'
                y_pred=model.predict(x,x_mask,n_batch)
                print y_pred
            if idx%clip_freq==0:
                print 'cliping learning rate:',
                lr=lr/2
                print lr

    print "Finished. Time = "+str(time.time()-start)


if __name__ == '__main__':
    train(lr=lr)