import time

from rnnlm import *
from utils import TextIterator,save_model

lr=0.01
p=0.5
n_batch=5
NEPOCH=100

n_input=100
n_hidden=250
maxlen=100
cell='gru'
optimizer='sgd'
train_datafile='../data/billion.tr'
test_datafile='../data/billion.te'
index2word_file='../data/index2word.pkl'
n_words_source=-1
vocabulary_size=793473

disp_freq=100
sample_freq=200
save_freq=5000

def train():
    # Load data
    print 'loading dataset...'
    train_data=TextIterator(train_datafile,index2word_file,n_words_source=n_words_source,n_batch=n_batch,maxlen=maxlen)
    test_data=TextIterator(test_datafile,index2word_file,n_words_source=n_words_source,n_batch=n_batch,maxlen=maxlen)

    print 'building model...'
    model=RNNLM(n_input,n_hidden,vocabulary_size,cell,optimizer,p)
    print 'training start...'
    start=time.time()
    for epoch in xrange(NEPOCH):
        error=0
        idx=0
        in_start=time.time()
        for x,x_mask,y,y_mask in train_data:
            idx+=1
            cost=model.train(x,x_mask,y,y_mask,n_batch,lr)
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
                save_model('./model/parameters_%.2f.pkl'%(time.time()-start),model)
            if idx % sample_freq==0:
                print 'Sampling....'
                y_pred=model.predict(x,x_mask,n_batch)
                print y_pred

    print "Finished. Time = "+str(time.time()-start)


if __name__ == '__main__':
    train()
