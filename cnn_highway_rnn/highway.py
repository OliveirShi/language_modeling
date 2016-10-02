import numpy as np
import theano
import theano.tensor as T 

class Highway:
	def __init__(self):
		pass
	def forward(self):
		t=T.nnet.sigmoid(T.dot(self.Wh,self.y)+self.bh)
		z=t*ReLU(T.dot(self.Wz,self.y)+self.bz)+(1-t)*self.y
		self.output=z