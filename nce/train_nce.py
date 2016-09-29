import os
import numpy as np
import time
from utils_nce import *
from datetime import datetime
from gru_nce import GRUTheano


LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "2000"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "48"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))
NEPOCH = int(os.environ.get("NEPOCH", "20"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./data/reddit-comments-2015.csv")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "25000"))

if not MODEL_OUTPUT_FILE:
  ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
  MODEL_OUTPUT_FILE = "GRU-%s-%s-%s-%s.dat" % (ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

# Load data
x_train, y_train, word_to_index, index_to_word, sorted_vocab = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)

k = 2*1000
alpha = 0
q_dis = Q_dis(word_to_index,sorted_vocab,alpha)
q_w = Q_w(word_to_index,sorted_vocab,alpha)
print q_dis
print q_w
# create Q distribution

# for each train_y create negative sampling vector index
# print y_train[0]
# neg_m = negative_sample(y_train[0],k,q_dis)
# print neg_m
# print neg_m.shape

# the neg_m shape (len(s) , k) !


# Build model
model = GRUTheano(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)

# Print SGD step time
t1 = time.time()
c_o_t = model.step_check(x_train[10], y_train[10], negative_sample(y_train[10],k,q_dis), q_w)
print c_o_t
print c_o_t.shape
t2 = time.time()
print "SGD One Step time: %f milliseconds" % ((t2 - t1) * 1000.)

# We do this every few examples to understand what's going on

train_with_sgd(model, x_train, y_train, k , q_dis, q_w,  learning_rate=LEARNING_RATE, nepoch=NEPOCH, decay=0.9)
