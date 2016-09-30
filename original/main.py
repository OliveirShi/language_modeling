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
        save_model('./model.rnnlm.model_'+str(i),model)
    print "Iter = "+str(i)+ ", Loss = "+str(error)+", Time = "+str(in_time)
    if error<e:
        break;
print "Finished. Time = "+str(time.time()-start)

print "save model..."
save_model("./model/rnnlm.model",model)


def train_with_sgd(model, X_train, y_train, k, q_dis, q_w, learning_rate=0.001, nepoch=20, decay=0.9,
    callback_every=10000, callback=None):
    num_examples_seen = 0
    for epoch in range(nepoch):
        # For each training example...
        print epoch
        for i in np.random.permutation(len(y_train)):
            # One SGD step
            cost=model.sgd_step(X_train[i], y_train[i], negative_sample(y_train[i],k,q_dis), q_w, learning_rate, decay)
            print cost,
            num_examples_seen += 1
    return model