import numpy as np
import theano
import theano.tensor as T

from softmax import *
from gru import *
from lstm import *
from updates import *

class RNNLM:
    def __init__(self,n_input,n_hidden,n_output,E,cell='gru',optimizer='sgd',p=0.5):
        self.x=T.tensor3('x')
        self.y=T.tensor3('y')
        
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.n_output=n_output
        self.E=E

        self.cell=cell
        self.p=p
        self.is_train=T.iscalar('is_train')
        self.n_batch=T.iscalar('n_batch')
        self.maskx=T.matrix('maskx')
        self.masky=T.matrix('masky')
        self.optimizer=optimizer
        self.epsilon=1.0e-15
        self.build()
        

    def build(self):
        rng=np.random.RandomState(1234)
        if self.cell=='gru':
            hidden_layer=GRU(self.rng,
                             self.n_input,self.n_hidden,
                             self.x,self.E,self.maskx,
                             self.is_train,self.p)
        else:
            hidden_layer=LSTM(self.rng,
                              self.n_input,self.n_hidden,
                              self.x,self.E,self.maskx,
                              self.is_train,self.p)

        output_layer=softmax(self.n_hidden,self.n_output,hidden_layer.activation,self.n_batch)
        self.params=self.E+hidden_layer.params+output_layer.params

        cost=self.categorial_cross_entropy(output_layer.activation,self.y)

   
        lr=T.scalar("lr")
        gparams=[T.clip(T.grad(cost,p),-10,10) for p in self.params]
        updates=sgd(self.params,gparams,lr)
        

        self.train=theano.function(inputs=[self.x,self.maskx,self.y,self.masky],
                                   output=cost,
                                   updates=updates,
                                   givens={self.is_train:np.case['int32'](1)})
        self.predict=theano.function(inputs=[self.x,self.maskx,self.n_batch],
                                     outputs=activation,
                                     givens={self.is_train:np.case['int32'](0)})

    def categorical_crossentropy(self,y_pred,y_true):
        y_pred=T.clip(y_pred,self.epsilon,1.0-self.epsilon)
        
        nll=T.nnet.categorial_crossentropy(y_pred,y_true)
        return T.sum(nll*self.masky)/T.sum(self.masky)
    
