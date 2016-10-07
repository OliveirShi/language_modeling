import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d,relu
from theano.tensor.signal import downsample

class ConvPool(object):
	"""Convolution + max_pool"""
	def __init__(self, rng,input,filters_shape,image_shape,pool_size=(2,2),non_linear="tanh"):
		
		self.rng = rng
		self.input = input
		self.filters_shape = filters_shape
		self.image_shape = image_shape
		self.pool_size = pool_size
		self.non_linear=non_linear

		init_W=np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                             high=np.sqrt(1./n_input),
                                             size=(n_hidden,n_input)))
		init_b=np.zeros((n_hidden),dtype=theano.config.floatX)

		self.W=theano.shared(value=init_W,name='W')
		self.b=theano.shared(value=init_b,name='b')

		self.params=[self.W,self.b]
		self.build()

	def build(self):
		# convolution
		conv_out=conv2d(input=input,
			filters=self.W,filters_shape=filters_shape,input_shape=image_shape)

		if self.non_linear=='tanh':
			conv_out_tanh=T.tanh(conv_out+self.b.dimshuffle('x',0,'x','x'))
			self.output=downsample.max_pool_2d(input=conv_out_tanh,ds=self.pool_size,ignore_border=True)
		elif self.non_linear=='relu':
			conv_out_relu=relu(conv_out+self.b.dimshuffle('x',0,'x','x'))
			self.output=downsample.max_pool_2d(input=conv_out_relu,ds=self.pool_size,ignore_border=True)
		else:
			pooled_out=downsample.max_pool_2d(input=conv_out,ds=self.pool_size,ignore_border=True)
			self.output=pooled_out+self.b.dimshuffle('x',0,'x','x')




