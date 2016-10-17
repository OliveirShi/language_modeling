import cPickle as pickle
def build_vocab(filename):
    fr=open(filename,'r').read().split('\n')
    vocab=dict()

    for line in fr:
        words=line.split(' ')
        for w in words:
            if len(w)>=1 and w not in vocab:
                vocab[w]=1
            elif len(w)>=1:
                vocab[w]+=1
    vocab_freq=sorted(vocab.items(),cmp=lambda x,y:cmp(x[1],y[1]),reverse=True)
    vocab_dict={}
    vocab_dict['<s>']=0
    vocab_dict['</s>']=1
    index=2
    for item in vocab_freq:
        vocab_dict[item[0]]=index
        index+=1
        print item[0],vocab_dict[item[0]]
    print len(vocab_dict)
    with open('vocab_dict.pkl','w')as f:
        pickle.dump(vocab_dict,f)
    with open('vocab_freq.pkl','w')as f:
        pickle.dump(vocab_freq,f)

def word2index(filename):
    fr=open(filename,'r').read().split('\n')
    fw=open('idx_'+filename,'w')
    with open('vocab_dict.pkl','r')as f:
        vocab_dict=pickle.load(f)
    for line in fr:
        words=line.split(' ')
        fw.write('0 ')
        for w in words:
            if len(w)>=1:
                fw.write(str(vocab_dict[w])+' ')
        fw.write('1\n')
    fw.flush()
    fw.close()
build_vocab('ptb.train.txt')


