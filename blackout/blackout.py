from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
import theano.tensor as T
import numpy as np
srng = RandomStreams(seed=234)



i = T.iscalar('i')

im = np.zeros((15,2))
dis = [[i]*i for i in range(1,10)]
q_dis = []
for d in dis:
    q_dis.extend(d)
print q_dis



rv_u = srng.random_integers((1,5),low=0,high=1)

rv_n = srng.normal((2,2))

wm = np.random.uniform(-1,1,(15,3))
w = T.matrix('w')
print w

mask_w = w[rv_u[0]]

f = function([w], [rv_u,mask_w])
g = function([], rv_n, no_default_updates=True)    #Not updating rv_n.rng
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)

print f(wm)
print g()
