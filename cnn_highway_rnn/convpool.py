import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d,relu
from theano.tensor.signal import downsample

class ConvPool(object):
    """Convolution + max_pool"""
    def __init__(self, rng,input,filter_shape,image_shape,pool_size=(2,2),non_linear="tanh"):

        self.rng = rng
        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.pool_size = pool_size
        self.non_linear=non_linear

        fan_in=np.prod(filter_shape[1:])
        fan_out=(filter_shape[0]*np.prod(filter_shape[2:])/np.prod(pool_size))

        if self.non_linear=='none'or self.non_linear=='relu':
            init_W=np.asarray(np.random.uniform(low=-0.01,
                                             high=0.01,
                                             size=filter_shape),
                              dtype=theano.config.floatX)
            self.W=theano.shared(value=init_W,name='W_conv')
        else:
            init_W=np.asarray(np.random.uniform(low=-np.sqrt(1./(fan_in+fan_out)),
                                                high=np.sqrt(1./(fan_in+fan_out)),
                                                size=filter_shape),
                              dtype=theano.config.floatX)
            self.W=theano.shared(value=init_W,name='W_conv')
        init_b=np.zeros((filter_shape[0],),dtype=theano.config.floatX)
        self.b=theano.shared(value=init_b,name='b_conv')


        self.params=[self.W,self.b]
        self.build()

    def build(self):
        # convolve input feawture maps with filters
        conv_out=conv2d(input=input,
			filters=self.W,filters_shape=self.filter_shape,image_shape=self.image_shape)

        if self.non_linear=='tanh':
            conv_out_tanh=T.tanh(conv_out+self.b.dimshuffle('x',0,'x','x'))
            self.output=downsample.max_pool_2d(input=conv_out_tanh,ds=self.pool_size,ignore_border=True)
        elif self.non_linear=='relu':
            conv_out_relu=relu(conv_out+self.b.dimshuffle('x',0,'x','x'))
            self.output=downsample.max_pool_2d(input=conv_out_relu,ds=self.pool_size,ignore_border=True)
        else:
            pooled_out=downsample.max_pool_2d(input=conv_out,ds=self.pool_size,ignore_border=True)
            self.output=pooled_out+self.b.dimshuffle('x',0,'x','x')





