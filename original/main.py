import time
import sys
import os

import numpy as np
import theano
import theano.tensor as T

from rnn import *
from utils import *

lr=0.01
p=0.5
n_batch=10

n_input=300
n_hidden=500
n_output=80000
cell='gru'
optimzer='sgd'


model=RNNLM(n_input,n_hidden,n_output,cell,optimizer,p)
print 'training...'
start=time.time()
g_error=999.99
for i in xrange(n_epoch):
    error=0
    in_start=time.time()
    for batch_id,xy in data_xy.items():
        cost=model.train(x,maskx,y,masky,lr)
        error+=cost
        print i,

    error/=len(seqs)
    if error<g_error:
        g_error=error
        save_model('./model.rnnlm.model_'+stri(i),model)
    print "Iter = "+str(i)+ ", Loss = "+str(error)+", Time = "+str(in_time)
    if error<e:
        break;
print "Finished. Time = "+str(time.time()-start)

print "save model..."
save_model("./model/rnnlm.model",model)
