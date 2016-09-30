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
input_data_file='data/reddit-comments-2015.csv'
vocabulary_size=1000

# Load data
x_train, y_train, word_to_index, index_to_word, sorted_vocab = load_data(input_data_file, vocabulary_size)


model=RNNLM(n_input,n_hidden,n_output,cell,optimizer,p)
print 'training...'
start=time.time()
g_error=999.99
for epoch in xrange(NEPOCH):
    error=0
    in_start=time.time()
    for batch_id,xy in data_xy.items():
        cost=model.train(x,maskx,y,masky,lr)
        error+=cost
        print epoch,

    error/=len(seqs)
    if error<g_error:
        g_error=error
        save_model('./model.rnnlm.model_'+str(epoch),model)

    print "Iter = "+str(epoch)+ ", Loss = "+str(error)+", Time = "+str(time.time()-in_start)
    if error<1e-10:
        break;
print "Finished. Time = "+str(time.time()-start)

print "save model..."
save_model("./model/rnnlm.model",model)


