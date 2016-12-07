import time
import numpy as np
import theano
import theano.tensor as T

def matrix(N=2):
    A=theano.shared(np.random.randn(1024,N*1024).astype(theano.config.floatX),name="A",borrow=True)
    B=theano.shared(np.random.randn(N*1024,1024).astype(theano.config.floatX),name="B",borrow=True)
    C=T.dot(A,B)

    model=theano.function([],C)

    tmin=0
    for _ in range(20):
	    t0=time.time()
	    model()
	    t1=time.time()
	    tmin+=(t1-t0)

    return tmin*1.0/10

for index in range(2,100):
    print index,matrix(index)


