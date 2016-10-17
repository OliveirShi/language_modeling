import numpy as np 
import theano
import theano.tensor as T
from convpool import ConvPool 
from highway import Highway
from lstm import LSTM 
from gru import GRU 

class cnn_highway_lstm_lm:
	def __init(self):
		self.build()
	def build(self):
		self.char_layer=ConvPool()
		self.input_layer=Highway()
		self.hidden_layer=LSTM()
		