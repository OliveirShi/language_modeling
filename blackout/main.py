import time
from utils import *
from grulm import GRULM

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
word2index_file='../data/index2word.pkl'
vocab_file='../data/vocab.txt'
n_words_source=-1
vocabulary_size=793473

disp_freq=100
sample_freq=200
save_freq=5000


k = 2*1000
alpha = 0.75

with open(word2index_file,'r')as f:
    word2index=pickle.load(f)
vocab=open(vocab_file,'r').read().split('\n')

q_dis = Q_dis(word2index,vocab,alpha)
q_w = Q_w(word2index,vocab,alpha)
print q_dis
print q_w
# create Q distribution

# for each train_y create negative sampling vector index
# print y_train[0]
# neg_m = negative_sample(y_train[0],k,q_dis)
# print neg_m
# print neg_m.shape



def train():
    # Load data
    print 'loading dataset...'
    train_data=TextIterator(train_datafile,n_words_source=n_words_source,maxlen=maxlen)
    test_data=TextIterator(test_datafile,n_words_source=n_words_source,maxlen=maxlen)

    print 'building model...'
    model=GRULM(n_input,n_hidden,vocabulary_size)
    print 'training start...'
    start=time.time()
    for epoch in xrange(NEPOCH):
        error=0
        idx=0
        in_start=time.time()
        for x,y in train_data:
            idx+=1
            cost=model.train(x, y, negative_sample(y,k,q_dis), q_w,lr)
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
