import numpy as np
import theano
import theano.tensor as T 

class Highway:
    def __init__(self,n_input,n_hidden,x):

        self.x=x
        # hidden
        init_Wh=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_hidden,n_input+n_hidden)))
        init_bh=np.zeros((n_hidden),dtype=theano.config.floatX)

        self.Wh=theano.shared(value=init_Wh,name='Wh')
        self.bh=theano.shared(value=init_bh,name='bh')

        # carry
        init_Wc=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_hidden,n_input+n_hidden)))
        init_bc=np.zeros((n_hidden),dtype=theano.config.floatX)

        self.Wc=theano.shared(value=init_Wc,name='Wc')
        self.bc=theano.shared(value=init_bc,name='bc')

        # gate
        init_Wt=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_hidden,n_input+n_hidden)))
        init_bt=np.zeros((n_hidden),dtype=theano.config.floatX)

        self.Wt=theano.shared(value=init_Wt,name='Wt')
        self.bt=theano.shared(value=init_bt,name='bt')
        self.build()
        self.params=[self.Wh,self.Wc,self.Wt,self.bh,self.bc,self.bt]

    def build(self):
        hidden=T.nnet.sigmoid(T.dot(self.x,self.Wh)+self.bh)
        carry=T.tanh(T.dot(hidden,self.Wc)+self.bc)
        gate=T.nnet.relu(T.dot(hidden,self.Wt)+self.bt)
        self.activation=gate*carry+(1-carry)*hidden
