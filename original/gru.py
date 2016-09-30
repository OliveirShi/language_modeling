import numpy as np
import theano
import theano.tensor as T

class GRU:
    def __init__(self,rng,
                 n_input,n_hidden,
                 x,E,mask,
                 is_train=1,p=0.5):
        self.rng=rng

        self.n_input=n_input
        self.n_hidden=n_hidden
        self.f=T.nnet.sigmoid

        self.x=x
        self.E=E
        self.mask=mask
        self.p=p

        # Update gate
        init_Wz=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_hidden,n_input+n_hidden)))
        init_bz=np.zeros((n_hidden),dtype=theano.config.floatX)

        self.Wz=theano.shared(value=init_Wz,name='Wz')
        self.bz=theano.shared(value=init_bz,name='bz')

        # Reset gate
        init_Wr=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_hidden,n_input+n_hidden)))
        init_br=np.zeros((n_hidden),dtype=theano.config.floatX)

        self.Wr=theano.shared(value=init_Wr,name='Wr')
        self.br=theano.shared(value=init_br,name='br')

        # Cell update
        init_Wxc=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_hidden,n_input)))
        init_Whc=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_hidden,n_hidden)))
        init_bc=np.zeros((n_hidden),dtype=theano.config.floatX)

        self.Wxc=theano.shared(value=init_Wxc,name='Wxc')
        self.Whc=theano.shared(value=init_Whc,name='Whx')
        self.bc=theano.shared(value=init_bc,name='bc')

        # Params
        self.params=[self.Wz,self.Wr,self.bz,self.br]

        self.build()

    def build(self):

        def _recurrence(x_t,m,h_tm1):
            x_e=self.E[:,x_t]
            concated=T.concatenated([x_e,h_tm1])

            # Update gate
            z_t=self.f(T.dot(self.Wz, concated) + self.bz )
            # Input fate
            r_t=self.f(T.dot(self.Wr, concated) + self.br )

            # Cell update
            c_t=T.tanh(T.dot(self.Wxc,x_e)+T.dot(self.Whc,r_t*h_tm1)+self.bc)

            h_t=(T.ones_like(z_t)-z_t) * c_t + z_t * h_tm1

            h_t=h_t*m[:,None]


            return h_t

        h,_=theano.scan(fn=_recurrence,
                            sequences=[self.x,self.mask],
                            truncate_gradient=-1,
                            outputs_info=[dict(init=T.zeros(self.n_hidden))])

        # Dropout
        if self.p>0:
            srng=T.shared_randomstreams.RandomStreams(self.rng.randint(99999))
            drop_mask=srng.binomial(n=1,p=1-self.p,size=h.shape,dtype=theano.config.floatX)
            self.activation=T.switch(T.eq(self.is_train,1),h*drop_mask,h*(1-p))
        else:
            self.activation=T.switch(T.eq(self.is_train,1),h,h)
            
                
        
