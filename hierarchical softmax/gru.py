import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class GRULayer(object):
    def __init__(self,X,mask,shape,is_train=1,p=0.5,state_pre=None):
        prefix="GRU"
        self.in_size,self.hidden_size=shape

        self.W_xr=theano.shared(value=np.asarray((np.random.randn(self.in_size,self.hidden_size) * 0.1),dtype=theano.config.floatX),
                                name=prefix+'_W_xr')
        self.W_hr=theano.shared(value=np.asarray((np.random.randn(self.hidden_size,self.hidden_size) * 0.1),dtype=theano.config.floatX),
                                name=prefix+'_W_hr')
        self.b_r=theano.shared(value=np.asarray(np.zeros(self.hidden_size),dtype=theano.config.floatX),
                               name=prefix+'_b_r')

        self.W_xz=theano.shared(value=np.asarray((np.random.randn(self.in_size,self.hidden_size) * 0.1),dtype=theano.config.floatX),
                                name=prefix+'_W_xz')
        self.W_hz=theano.shared(value=np.asarray((np.random.randn(self.hidden_size,self.hidden_size) * 0.1),dtype=theano.config.floatX),
                                name=prefix+'_W_hz')
        self.b_z=theano.shared(value=np.asarray(np.zeros(self.hidden_size),dtype=theano.config.floatX),
                               name=prefix+'_b_z')

        self.W_xh=theano.shared(value=np.asarray((np.random.randn(self.in_size,self.hidden_size) * 0.1),dtype=theano.config.floatX),
                                name=prefix+'_W_xh')
        self.W_hh=theano.shared(value=np.asarray((np.random.randn(self.hidden_size,self.hidden_size) * 0.1),dtype=theano.config.floatX),
                                name=prefix+'_W_hh')
        self.b_h=theano.shared(value=np.asarray(np.zeros(self.hidden_size),dtype=theano.config.floatX),
                               name=prefix+'_b_h')

        self.X=X
        self.mask=mask


        batch_size=self.X.shape[1]
        if state_pre==None:
            state_pre=T.zeros((batch_size,self.hidden_size),dtype=theano.config.floatX)

        def _step(x,m,h_tm1):
            r=T.nnet.sigmoid(T.dot(x,self.W_xr) + T.dot(h_tm1,self.W_hr) +self.b_r)
            z=T.nnet.sigmoid(T.dot(x,self.W_xz) + T.dot(h_tm1,self.W_hz) +self.b_z)

            gh=T.tanh(T.dot(x , self.W_xh) + T.dot(r * h_tm1 , self.W_hh) + self.b_h)

            h_t=z * h_tm1 + (T.ones_like(z) - z) * gh

            h_t = h_t * m[:,None]

            return h_t

        h,_=theano.scan(fn=_step,
                        sequences=[self.X,self.mask],
                        outputs_info=state_pre)
        self.h=h
        if p>0:
            trng=RandomStreams(12345)
            drop_mask=trng.binomial(n=1,p=1-p,size=h.shape,dtype=theano.config.floatX)
            self.activation=T.switch(T.eq(is_train,1),h*drop_mask,h*(1-p))
        else:
            self.activation=T.switch(T.eq(is_train,1),h,h)

        self.params=[self.W_xr,self.W_hr,self.b_r,
                     self.W_xz,self.W_hz,self.b_z,
                     self.W_xh,self.W_hh,self.b_h]