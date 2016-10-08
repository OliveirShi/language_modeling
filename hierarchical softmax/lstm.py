import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class LSTMLayer(object):
    def __init__(self,rng,layer_id,shape,X,mask,is_train=1,batch_size=1,p=0.5):
        prefix="LSTM"
        layer_id='_'+layer_id
        self.in_size,self.out_size=shape

        self.W_xi=theano.shared(value=np.asarray((np.random.randn(self.in_size,self.out_size)*0.1),dtype=theano.config.floatX),
                                name=prefix+layer_id+'_W_xi')
        self.W_hi=theano.shared(value=np.asarray((np.random.randn(self.out_size,self.out_size)*0.1),dtype=theano.config.floatX),
                                name=prefix+layer_id+"_W_hi")
        self.W_ci=theano.shared(value=np.asarray((np.random.randn(self.out_size,self.out_size)*0.1),dtype=theano.config.floatX),
                                name=prefix+layer_id+"_W_ci")
        self.b_i=theano.shared(value=np.asarray(np.zeros(self.out_size),dtype=theano.config.floatX),
                               name=prefix+layer_id+'_b_i')

        self.W_xf=theano.shared(value=np.asarray((np.random.randn(self.in_size,self.out_size)*0.1),dtype=theano.config.floatX),
                                name=prefix+layer_id+'_W_xf')
        self.W_hf=theano.shared(value=np.asarray((np.random.randn(self.out_size,self.out_size)*0.1),dtype=theano.config.floatX),
                                name=prefix+layer_id+"_W_hf")
        self.W_cf=theano.shared(value=np.asarray((np.random.randn(self.out_size,self.out_size)*0.1),dtype=theano.config.floatX),
                                name=prefix+layer_id+"_W_cf")
        self.b_f=theano.shared(value=np.asarray(np.zeros(self.out_size),dtype=theano.config.floatX),
                               name=prefix+layer_id+'_b_f')

        self.W_xc=theano.shared(value=np.asarray((np.random.randn(self.in_size,self.out_size)*0.1),dtype=theano.config.floatX),
                                name=prefix+layer_id+'_W_xc')
        self.W_hc=theano.shared(value=np.asarray((np.random.randn(self.out_size,self.out_size)*0.1),dtype=theano.config.floatX),
                                name=prefix+layer_id+"_W_hc")
        self.b_c=theano.shared(value=np.asarray(np.zeros(self.out_size),dtype=theano.config.floatX),
                               name=prefix+layer_id+'_b_c')

        self.W_xo=theano.shared(value=np.asarray((np.random.randn(self.in_size,self.out_size)*0.1),dtype=theano.config.floatX),
                                name=prefix+layer_id+'_W_xo')
        self.W_ho=theano.shared(value=np.asarray((np.random.randn(self.out_size,self.out_size)*0.1),dtype=theano.config.floatX),
                                name=prefix+layer_id+"_W_ho")
        self.W_co=theano.shared(value=np.asarray((np.random.randn(self.out_size,self.out_size)*0.1),dtype=theano.config.floatX),
                                name=prefix+layer_id+"_W_co")
        self.b_o=theano.shared(value=np.asarray(np.zeros(self.out_size),dtype=theano.config.floatX),
                               name=prefix+layer_id+'_b_o')

        self.X=X
        self.mask=mask

        def _step(x,m,h_tm1,c_tm1):
            i_t=T.nnet.sigmoid(T.dot(x, self.W_xi) + T.dot(h_tm1, self.W_hi) + T.dot(c_tm1, self.W_ci) + self.b_i)
            f_t=T.nnet.sigmoid(T.dot(x, self.W_xf) + T.dot(h_tm1, self.W_hf) + T.dot(c_tm1, self.W_cf) + self.b_f)
            o_t=T.nnet.sigmoid(T.dot(x, self.W_xo) + T.dot(h_tm1, self.W_ho) + T.dot(c_tm1, self.W_co) + self.b_o)

            gc=T.tanh(T.dot(x,self.W_xc) + T.dot(h_tm1, self.W_hc) +self.b_c)

            c_t=f_t * c_tm1 + i_t * gc
            h_t=o_t * T.tanh(c_t)
            c_t=c_t * m[:,None]
            h_t=h_t * m[:None]
            return h_t,c_t

        [h,c],_=theano.scan(
            fn=_step,
            sequences=[self.X,self.mask],
            outputs_info=[T.alloc(np.asarray(0.,dtype=theano.config.floatX), 1, batch_size * self.out_size),
                          T.alloc(np.asarray(0.,dtype=theano.config.floatX), 1, batch_size * self.out_size)]
        )

        if p>0:
            srng=RandomStreams(12345)
            drop_mask=srng.binomial(n=1,p=1-p,size=h.shape,dtype=theano.config.floatX)
            self.activation=T.switch(T.eq(is_train,1),h*drop_mask,h*(1-p))
        else:
            self.activation=T.switch(T.eq(is_train,1),h,h)

        self.params=[self.W_xi, self.W_hi, self.W_ci, self.b_i,
                      self.W_xf, self.W_hf, self.W_cf, self.b_f,
                      self.W_xo, self.W_ho, self.W_co, self.b_o,
                      self.W_xc, self.W_hc,            self.b_c]




