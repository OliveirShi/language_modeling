import numpy as np
import cPickle as pickle
import Queue

def save_model(f,model):
    ps={}
    for p in model.params:
        ps[p.name]=p.get_value()
    pickle.dump(ps,open(f,'wb'))

def load_model(f,model):
    ps=pickle.load(open(f,'rb'))
    for p in model.params:
        p.set_value(ps[p.name])
    return model

class TextIterator:
    def __init__(self,source,freqs,n_batch,maxlen,n_words_source=-1):

        self.source=open(source,'r')
        self.nodes,self.choices=load_prefix(freqs)


        self.n_batch=n_batch
        self.maxlen=maxlen
        self.n_words_source=n_words_source
        self.end_of_data=False



    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data=False
            self.reset()
            raise StopIteration
        source=[]
        try:
            while True:
                s=self.source.readline()
                if s=="":
                    raise IOError
                s=s.strip().split(' ')

                if self.n_words_source>0:
                    s=[int(w) if int(w) <self.n_words_source else 3 for w in s]
                # filter long sentences
                if len(s)>self.maxlen:
                    continue
                source.append(s)
                if len(source)>=self.n_batch:
                    break
        except IOError:
            self.end_of_data=True

        if len(source)<=0:
            self.end_of_data=False
            self.reset()
            raise StopIteration
        return prepare_data(source)

def prepare_data(seqs_x):
    lengths_x=[len(s)-1 for s in seqs_x]
    n_samples=len(seqs_x)
    maxlen_x=np.max(lengths_x)

    x=np.zeros((maxlen_x,n_samples)).astype('int32')
    y=np.zeros((maxlen_x,n_samples)).astype('int32')
    x_mask=np.zeros((maxlen_x,n_samples)).astype('float32')
    y_mask=np.zeros((maxlen_x,n_samples)).astype('float32')

    for idx,s_x in enumerate(seqs_x):
        x[:lengths_x[idx],idx]=s_x[:-1]
        y[:lengths_x[idx],idx]=s_x[1:]
        x_mask[:lengths_x[idx],idx]=1
        y_mask[:lengths_x[idx],idx]=1

    return x,x_mask,y,y_mask,self.choices[]



class Node(object):
    def __init__(self,left=None,right=None,index=None):
        self.left=left
        self.right=right
        self.index=index

    def __repr__(self):
        string=str(self.index)
        if self.left:
            string+=', -1:'+str(self.left.index)
        if self.right:
            string+=', +1:'+str(self.right.index)
        return string

    def preorder(self,polarity=None,param=None,collector=None):
        if collector is None:
            collector=[]
        if polarity is None:
            polarity=[]
        if param is None:
            param=[]

        if self.left:
            if isinstance(self.left[1],Node):
                self.left[1].preorder(polarity+[-1],param+[self.index],collector)
            else:
                collector.append((self.left[1],param+[self.index], polarity + [-1]))
        if self.right:
            if isinstance(self.right[1],Node):
                self.right[1].preorder(polarity+[1],param+[self.index],collector)
            else:
                collector.append((self.right[1],param+[self.index], polarity + [1]))
        return collector


def build_huffman(frequenties):
    Q=Queue.PriorityQueue()
    for v in frequenties:  #((freq,word),index)
        Q.put(v)
    idx=0
    while Q.qsize()>1:
        l,r=Q.get(),Q.get()
        node=Node(l,r,idx)
        idx+=1
        freq=l[0]+r[0]
        Q.put((freq,node))
    return Q.get()[1]


def load_prefix(freq_file):
    rel_freq=pickle.load(open(freq_file, 'rb'))
    freq = zip(rel_freq, range(len(rel_freq)))
    tree = build_huffman(freq)
    x = tree.preorder()
    x=sorted(x, key=lambda z: z[0])
    nodes=[]
    choices=[]
    for idx,node,choice in x:
        nodes.append(np.asarray(node))
        choices.append(np.asarray(choice))
    return nodes,choices



