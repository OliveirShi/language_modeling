import time
import numpy as np
import theano
import theano.tensor as T

N=2
A=theano.shared(np.random.randn(N*1024,1024).astype(theano.config.floatX),name="A")
B=theano.shared(np.random.randn(N*1024,1024).astype(theano.config.floatX),name="B")
C=T.dot(A,B)

model=theano.function([],C)

tmin=0
for _ in range(10):
	t0=time.time()
	model()
	t1=time.time()
	tmin+=(t1-t0)
	print t1-t0

print tmin*1.0/10