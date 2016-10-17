import cPickle as pickle
def vocab2dict(filename='vocab.txt'):
	fr=open(filename).read().split('\n')
	word2index={}
	index2word={}
	vocab={}
	index=0
	for line in fr:
		word2index[line]=index
		index2word[index]=line
		index+=1
	with open('word2index.pkl','wb')as f:
		pickle.dump(word2index,f)
	with open('index2word.pkl','wb')as f:
		pickle.dump(index2word,f)

def get_vocab(raw_file='billion.tr',vocab_file='vocab.txt'):
	vocab=open(vocab_file,'r').read().split('\n')
	fr=open(raw_file,'r').read().split('\n')
	vocab_dict=dict()
	vocab_set=set()
	for word in vocab:
		vocab_set.add(word)
		vocab_dict[word]=0
	for line in fr:
		words=line.split(' ')
		for w in words:
			if w in vocab_set:
				vocab_dict[w]+=1
			else:
				vocab_dict['<UNK>']+=1
	vocab=[]
	for item in vocab_dict:
		vocab.append((item,vocab_dict[item]))
	with open('vocab.pkl','wb')as f:
		pickle.dump(vocab,f)
vocab2dict()
