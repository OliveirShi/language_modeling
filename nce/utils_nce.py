import csv
import itertools
from os.path import isfile
import numpy as np
import nltk
import sys
import cPickle as pickle
import operator
import theano
from gru_nce import GRUTheano

SENTENCE_START_TOKEN = "SENTENCE_START"
SENTENCE_END_TOKEN = "SENTENCE_END"
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"

def Q_dis(word_to_index,vocab,alpha):
    """
    create Q distribution for negative word sampling
    when alpha is 0:
    wc**alpha / sum(wc**alpha) -> uniform distrbution
    wc**1 / sum(wc**1) -> unigram distribution

    notice the q_dis do not include the <MASK/> and UNK
    """
    q_dis = []
    for item in vocab:
        tmp = [word_to_index[item[0]]]*int(item[1]**alpha)
        # e.g. 'this' 342 -> [ 'this' , 'this' ,..,'this']
        q_dis.extend(tmp)

    return np.asarray(q_dis)

def Q_w(word_to_index,vocab,alpha):
    """
    weight for blackout the 1/relative frequence of the word
    """
    q_w = np.ones(len(vocab))

    q_t = 0
    for item in vocab:
        q_t = q_t + float(item[1]**alpha)

    for item in vocab:
        q_w[word_to_index[item[0]]] = float(item[1]**alpha)/float(q_t)

    return np.asarray(q_w,dtype=theano.config.floatX)

def nce(q_dis,k,i):
    """
    sampling K negative word from q_dis, Sk != i
    """
    ne_sample = []

    while len(ne_sample) < k:
        p = np.random.randint(low=0,high=len(q_dis)-1,size=1)[0]
        if q_dis[p] == i:
            pass
        else:
            ne_sample.append(p)

    return np.asarray(ne_sample)


def negative_sample(y_train_i,k,q_dis):
    """
    negative sampling for integer vector y_train_i
    """
    neg_m = []
    for i in y_train_i:
        neg_m.append(nce(q_dis,k,i))

    return np.asarray(neg_m)


def load_data(filename="data/reddit-comments-2015-08.csv", vocabulary_size=2000, min_sent_characters=0):
    if isfile('data/dataset.pkl'):
        with open('data/dataset.pkl')as f:
            (X_train, y_train, word_to_index, index_to_word, sorted_vocab)=pickle.load(f)
            return X_train, y_train, word_to_index, index_to_word, sorted_vocab
    word_to_index = []
    index_to_word = []

    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print("Reading CSV file...")

    with open(filename, 'rt') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode("utf-8").lower()) for x in reader])
        # Filter sentences
        sentences = [s for s in sentences if len(s) >= min_sent_characters]
        sentences = [s for s in sentences if "http" not in s]
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (SENTENCE_START_TOKEN, x, SENTENCE_END_TOKEN) for x in sentences]
    print("Parsed %d sentences." % (len(sentences)))


    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)[:vocabulary_size-2]
    
    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    sorted_vocab = sorted(vocab, key=operator.itemgetter(1))

    # concern about blackout algorithm
    sorted_vocab.append(("<MASK/>",1))
    sorted_vocab.append((UNKNOWN_TOKEN,1))

    index_to_word = [x[0] for x in sorted_vocab]
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    with open('data/dataset.pkl','w')as f:
        pickle.dump((X_train, y_train, word_to_index, index_to_word, sorted_vocab),f)
    return X_train, y_train, word_to_index, index_to_word, sorted_vocab


def train_with_sgd(model, X_train, y_train, k, q_dis, q_w, learning_rate=0.001, nepoch=20, decay=0.9,
    callback_every=10000, callback=None):
    num_examples_seen = 0
    for epoch in range(nepoch):
        # For each training example...
        print epoch
        error=0
        index=0
        for i in np.random.permutation(len(y_train)):
            # One SGD step
            index+=1
            cost=model.sgd_step(X_train[i], y_train[i], negative_sample(y_train[i],k,q_dis), q_w, learning_rate, decay)
            error+=cost
            if index%20==0:
                print error*1.0/20,'\t',
                error=0
                index=0
            num_examples_seen += 1
    return model

def save_model_parameters_theano(model, outfile):
    np.savez(outfile,
        E=model.E.get_value(),
        U=model.U.get_value(),
        W=model.W.get_value(),
        V=model.V.get_value(),
        b=model.b.get_value(),
        c=model.c.get_value())
    print "Saved model parameters to %s." % outfile

def load_model_parameters_theano(path, modelClass=GRUTheano):
    npzfile = np.load(path)
    E, U, W, V, b, c = npzfile["E"], npzfile["U"], npzfile["W"], npzfile["V"], npzfile["b"], npzfile["c"]
    hidden_dim, word_dim = E.shape[0], E.shape[1]
    print "Building model model from %s with hidden_dim=%d word_dim=%d" % (path, hidden_dim, word_dim)
    sys.stdout.flush()
    model = modelClass(word_dim, hidden_dim=hidden_dim)
    model.E.set_value(E)
    model.U.set_value(U)
    model.W.set_value(W)
    model.V.set_value(V)
    model.b.set_value(b)
    model.c.set_value(c)
    return model


def print_sentence(s, index_to_word):
    sentence_str = [index_to_word[x] for x in s[1:-1]]
    print(" ".join(sentence_str))
    sys.stdout.flush()

def generate_sentence(model, index_to_word, word_to_index, min_length=5):
    # We start the sentence with the start token
    new_sentence = [word_to_index[SENTENCE_START_TOKEN]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[SENTENCE_END_TOKEN]:
        next_word_probs = model.predict(new_sentence)[-1]
        samples = np.random.multinomial(1, next_word_probs)
        sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
        # Seomtimes we get stuck if the sentence becomes too long, e.g. "........" :(
        # And: We don't want sentences with UNKNOWN_TOKEN's
        if len(new_sentence) > 100 or sampled_word == word_to_index[UNKNOWN_TOKEN]:
            return None
    if len(new_sentence) < min_length:
        return None
    return new_sentence

def generate_sentences(model, n, index_to_word, word_to_index):
    for i in range(n):
        sent = None
        while not sent:
            sent = generate_sentence(model, index_to_word, word_to_index)
        print_sentence(sent, index_to_word)

