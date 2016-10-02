import numpy as numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool


class ConvPool(object):
	"""docstring for ConvPool"""
	def __init__(self, rng,input,filters_shape,image_shape,pool_size=(2,2),non_linear="tanh"):
		
		self.rng = rng
		self.input = input
		self.filters_shape = filters_shape
		self.image_shape = image_shape
		self.pool_size = pool_size
		self.non_linear=non_linear

		assert image_shape[1]=filters_shape[1]
		fan_in=np.prod(filters_shape[1:])
		fan_out=(filters_shape[0]*np.prod(filters_shape[2:]))

		self.W=theano.shared(value=init_W,name='W')
		self.b=theano.shared(value=init_b,name='b')

		# convolve
		conv_out=conv2d(input=input,
			filters=self.W,filters_shape=filters_shape,input_shape=image_shape)

		if self.non_linear=='tanh':
			conv_out_tanh=T.tanh(conv_out+self.b.dimshuffle('x',0,'x','x'))
			self.output=downsample.max_pool_2d(input=conv_out_tanh,ds=self.pool_size,ignore_border=True)
		elif self.non_linear=='relu':
			conv_out_relu=ReLU(conv_out+self.b.dimshuffle('x',0,'x','x'))
			self.output=downsample.max_pool_2d(input=conv_out_relu,ds=self.pool_size,ignore_border=True)
		else:
			pooled_out=downsample.max_pool_2d(input=conv_out,ds=self.pool_size,ignore_border=True)
			self.output=pooled_out+self.b.dimshuffle('x',0,'x','x')

		self.params=[self.W,self.b]


