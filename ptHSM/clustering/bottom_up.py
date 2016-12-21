import numpy as np 
import theano 
import theano.tensor as T 

def bottom_up(word_vectors,word_labels):
	while(!word_vectors.isempty()){
		word=word_vectors.pop();
		min=9999
		index=0
		for(w in word_vectors){
			cos_dis=cos(word,w)
			if cos_dis<min{
				min=cos_dis
				index=w
			}
		}
		another_list.add(word,index)

	}
	return another_list

another_list=bottom_up()
while len(another_list)!=1:
	another_list=bottom_up(another_list)