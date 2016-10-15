import time
from utils import *
from grulm import GRULM

lr=0.01
p=0.5
n_batch=5
NEPOCH=100

n_input=30
n_hidden=20
maxlen=30
cell='gru'
optimizer='sgd'
train_datafile='../data/billion.tr'
test_datafile='../data/billion.te'
word2index_file='../data/word2index.pkl'
vocab_file='../data/vocab.pkl'
n_words_source=-1
vocabulary_size=793473

disp_freq=100
sample_freq=200
save_freq=5000


k = 10
alpha = 0.75


def train():
    with open(word2index_file,'r')as f:
        word2index=pickle.load(f)
    with open(vocab_file,'r') as f:
        vocab=pickle.load(f)
    vocab_p = Q_w(word2index,vocab,alpha)
    print 'vocab_probabilty:',vocab_p.shape

    # Load data
    print 'loading dataset...'
    train_data=TextIterator(train_datafile,n_words_source=n_words_source,maxlen=maxlen)
    test_data=TextIterator(test_datafile,n_words_source=n_words_source,maxlen=maxlen)

    print 'building model...'
    model=GRULM(n_hidden,vocabulary_size)
    print 'training start...'
    start=time.time()
    for epoch in xrange(NEPOCH):
        error=0
        idx=0
        in_start=time.time()
        for (x,y) in train_data:
            print 'x:',x.shape
            print x
            print 'y:',y.shape
            print y
            idx+=1
            negy=negative_sample(y,k,vocab_p)
            print 'negy:',negy.shape
            print negy
            cost=model.train(x, y, negy, vocab_p,lr)
            print 'index:',idx,'cost:',cost
            error+=np.sum(cost)
            if np.isnan(cost) or np.isinf(cost):
                print 'NaN Or Inf detected!'
                return -1
            if idx % disp_freq==0:
                print 'epoch:',epoch,'idx:',idx,'cost:',error/disp_freq
                error=0
            if idx%save_freq==0:
                print 'dumping...'
                save_model('model/parameters_%.2f.pkl'%(time.time()-start),model)
            if idx % sample_freq==0:
                print 'Sampling....'
                #y_pred=model.predict(x,x_mask,n_batch)
                #print y_pred

    print "Finished. Time = "+str(time.time()-start)


if __name__ == '__main__':
    train()
