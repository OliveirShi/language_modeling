import numpy as np
import theano
import theano.tensor as T

class softmax:
    def __init__(self,n_input,n_output,x):
        self.n_input=n_input
        self.n_output=n_output

        logit_shape=x.shape
        self.x=x.reshape([logit_shape[0]*logit_shape[1],logit_shape[2]])

        init_W=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_output,n_input)),dtype=theano.config.floatX)
        init_b=np.zeros((n_output),dtype=theano.config.floatX)

        self.W=theano.shared(value=init_W,name='W')
        self.b=theano.shared(value=init_b,name='b')

        self.params=[self.W,self.b]

        self.build()

    def build(self):

        self.activation=T.nnet.softmax(T.dot(self.x,self.W)+self.b)
        self.prediction=T.argmax(self.activation,axis=1)
        

        
        
