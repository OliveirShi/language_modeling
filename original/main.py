import time

from rnnlm import *
from utils import TextIterator,save_model

import logging
from argparse import ArgumentParser 

lr=0.5
p=0
n_batch=2
NEPOCH=200

n_input=200
n_hidden=300
cell='gru'
optimizer='sgd'

p = ArgumentParser(usage='it is usage tip', description='no')  
p.add_argument('--train_file', default='../data/ptb/idx_ptb.train.txt', type=str, help='train dir')  
p.add_argument('--valid_file', default='../data/ptb/idx_ptb.valid.txt', type=str, help='valid dir')
p.add_argument('--test_file', default='../data/ptb/idx_ptb.test.txt', type=str, help='test dir')
p.add_argument('--vocab_size', default=10001, type=int, help='vocab size')

args = p.parse_args()  


train_datafile=str(args.train_file)
valid_datafile=str(args.valid_file)
test_datafile=str(args.test_file)
n_words_source=-1
vocabulary_size=int(args.vocab_size)
disp_freq=20
test_freq=200
save_freq=20000
clip_freq=2000
pred_freq=200

def evaluate(test_data,model):
    cost=0
    index=0
    for x,x_mask,y,y_mask in test_data:
        index+=1
        cost+=model.test(x,x_mask,y,y_mask,x.shape[1])
    return cost/index

def train(lr):
    # Load data
    print 'loading dataset...'
    train_data=TextIterator(train_datafile,n_words_source=n_words_source,n_batch=n_batch)
    test_data=TextIterator(test_datafile,n_words_source=n_words_source,n_batch=n_batch)

    print 'building model...'
    model=RNNLM(n_input,n_hidden,vocabulary_size,cell,optimizer,p)
    print 'training start...'
    start=time.time()
    idx=0
    for epoch in xrange(NEPOCH):
        error=0
        in_start=time.time()
        for x,x_mask,y,y_mask in train_data:
            idx+=1
            beg_time=time.time()
            cost=model.train(x,x_mask,y,y_mask,x.shape[1],lr)
            error+=np.sum(cost)
            if np.isnan(cost) or np.isinf(cost):
                print 'NaN Or Inf detected!'
                return -1
            if idx % disp_freq==0:
                print 'epoch:',epoch,'idx:',idx,'cost:',error/disp_freq,'base exp:',np.exp(error/disp_freq),'lr:',lr
                error=0
            if idx%save_freq==0:
                print 'dumping...'
                save_model('./model/parameters_%.2f.pkl'%(time.time()-start),model)
            if idx % test_freq==0:
                print 'testing....'
                test_cost=evaluate(test_data,model)
                print 'test_cost:',test_cost,'perplexity:',np.exp(test_cost)
            if idx % pred_freq==0:
                print 'predicting...'
                prediction=model.predict(x,x_mask,x.shape[1])
                print prediction[:100]
            if idx%clip_freq==0 and lr >=1e-3:
                print 'cliping learning rate:',
                lr=lr*0.7
                print lr

    print "Finished. Time = "+str(time.time()-start)


if __name__ == '__main__':
    train(lr=lr)
