import numpy as np
import theano
import theano.tensor as T
import Queue
import cPickle as pickle

class Softmaxlayer(object):
    def __init__(self,X,y,maskY,shape):
        prefix="n_softmax_"
        self.in_size,self.out_size=shape
        self.W=theano.shared(np.asarray((np.random.randn(shape) * 0.1),dtype=theano.config.floatX), prefix+'W')
        self.b=theano.shared(np.asarray(np.zeros(self.out_size),dtype=theano.config.floatX),prefix+'b')

        self.X=X
        self.y=y
        self.params=[self.W,self.b]

        def _step(x):
            y_pred=T.nnet.softmax( T.dot(x,self.W) + self.b )
            return y_pred

        y_pred,_=theano.scan(fn=_step,sequences=self.X)
        self.activation=T.nnet.categorical_crossentropy(y,y_pred*maskY)

class TreeNode(object):
    def __init__(self,index=None,left=None,right=None,parent=None,parent_choice=None):
        self.index=index
        self.right=right
        self.left=left
        self.parent=parent
        self.parent_choice=parent_choice

    def __repr__(self):
        return '<'+str(self.index)+', 0:'+str(self.left.index)+', 1:'+str(self.right.index)+'>'

class RessultNode(object):
    def __init__(self,value=None,parent=None):
        self.value=value
        self.parent=parent
        self.index='res:'+str(self.value)

    def __repr__(self):
        return '<'+str(self.value)+'>'




class HuffmanNode(object):
    def __init__(self,left=None,right=None,root=None):
        self.left=left
        self.right=right

    def children(self):
        return self.left,self.right

    def preorder(self,path=None,collector=None):
        if collector is None:
            collector=[]
        if path is None:
            path=[]

        if self.left:
            if isinstance(self.left[1],HuffmanNode):
                self.left[1].preorder(path+[-1],collector)
            else:
                collector.append((self.left[1],self.left[0],path+[-1]))
        if self.right:
            if isinstance(self.right[1],HuffmanNode):
                self.right[1].preorder(path+[1],collector)
            else:
                collector.append((self.right[1],self.right[0],path+[1]))
        return collector

def pad_bitstr(bitstr):
    """
    :param bitstr:
    :type bitstr: list
    :return: padded list of bits
    """
    max_bit_len = 0
    for bits in bitstr:
        if len(bits) > max_bit_len:
            max_bit_len = len(bits)
    for bits in bitstr:
        bits.extend([0] * (max_bit_len-len(bits)))

    return bitstr


def pad_virtual_class(clses, pad_value):
    max_cls_len = 0
    for nodes in clses:
        if len(nodes) > max_cls_len:
            max_cls_len = len(nodes)
    for nodes in clses:
        nodes.extend([pad_value] * (max_cls_len-len(nodes)))

    return clses

def build_huffman_tree(frequenties):
    Q=Queue.PriorityQueue()
    for v in frequenties:
        Q.put(v)
    while Q.qsize()>1:
        l,r=Q.get(),Q.get()
        node=HuffmanNode(l,r)
        Q.put((l[0]+r[0],node))
    return Q.get()

def prefix_generator(s, start=0, end=None):
    if end is None:
        end = len(s) + 1
    for idx in range(start, end):
        yield s[:idx]

def load_huffman_tree(rel_freq,meta_file=None):

    #with file(meta_file, 'rb') as f:
    #    meta = pickle.load(f)
    #    rel_freq = meta['rel_freq']
    freq = zip(rel_freq, range(len(rel_freq)))
    tree = build_huffman_tree(freq)[1]
    x = tree.preorder()
    y = sorted(x, key=lambda z: z[1], reverse=True)
    bitstr = []
    for _, _, bitstr in y:
        bitstr.append(bitstr[:-1])

    z=[]
    for wordindex,_,bits in y:
        z.append((wordindex,bits,list(prefix_generator(bits,end=len(bits)))))

    clses = set()
    for _, _, ele in z:
        for i in ele:
            clses.add(''.join('%+d' % j for j in i))
    idx2clses = sorted(clses, key=lambda ele: len(ele))
    cls2idx = dict(((cls, idx) for idx, cls in enumerate(idx2clses)))
    w = map(lambda x: (x[0], x[1], [cls2idx[''.join('%+d' % j for j in p)] for p in x[2]]), z)

    tmp1, tmp2 = [], []
    for _, bits, cls_idx in w:
        tmp1.append(bits)
        tmp2.append(cls_idx)
    pad_bitstr(tmp1)
    pad_virtual_class(tmp2, pad_value=len(idx2clses))
    assert len(freq) == len(w)
    idx2cls = [None] * len(freq)
    idx2bitstr = [None] * len(freq)
    for idx, bitstr_, cls_ in w:
        idx2cls[idx] = cls_
        idx2bitstr[idx] = bitstr_

    idx2cls = np.array(idx2cls, dtype='int32')
    idx2bitstr = np.array(idx2bitstr, dtype='int8')

    return idx2cls, idx2bitstr, idx2bitstr != 0



if __name__ == '__main__':
    freq = [
        (8.167, 'a'), (1.492, 'b'), (2.782, 'c'), (4.253, 'd'),
        (12.702, 'e'),(2.228, 'f'), (2.015, 'g'), (6.094, 'h'),
        (6.966, 'i'), (0.153, 'j'), (0.747, 'k'), (4.025, 'l'),
        (2.406, 'm'), (6.749, 'n'), (7.507, 'o'), (1.929, 'p'),
        (0.095, 'q'), (5.987, 'r'), (6.327, 's'), (9.056, 't'),
        (2.758, 'u'), (1.037, 'v'), (2.365, 'w'), (0.150, 'x'),
        (1.974, 'y'), (0.074, 'z')]
    load_huffman_tree(freq)


def prefix_generator(s, start=0, end=None):
    if end is None:
        end = len(s) + 1
    for idx in range(start, end):
        yield s[:idx]


class H_Softmax(object):

    def __init__(self,shape,
                 x,y,maskY):
        self.prefix='h_softmax_'

        self.in_size,self.out_size=shape
        # in_size:size,mb_size=out_size
        self.x=x
        self.y=y
        self.maskY=maskY
        self.rng=np.random.RandomState(12345)
        self.tree=build_binary_tree(range(self.out_size))
        # Make route
        self.build_route()
        self.build_graph()
        self.build_predict()

    def build_route(self):
        self.nodes=[]
        self.node_dict={}
        self.result_dict={}
        self.routes=[]

        self.label_count=0
        self.node_count=0
        for layer in self.tree:
            for node in layer:
                if isinstance(node,TreeNode): # middle units
                    self.node_count+=1
                    self.nodes.append(node)
                elif isinstance(node,RessultNode): # leaf untis
                    self.label_count+=1
                    self.result_dict[node.value]=node

        # Let's also put the tree into a matrix
        tree_matrix_val=np.ones((self.node_count+self.label_count,4),dtype=np.int32)* -1

        '''
        0: left tree node index
        1: right tree node index
        2: left leaf value
        3: right leaf value
        '''
        for layer in self.tree[::-1]:
            for node in layer:
                if isinstance(node,TreeNode):
                    try:
                        if not isinstance(node.left.index,str):
                            tree_matrix_val[node.index][0]=node.left.index
                        else:
                            tree_matrix_val[node.index][0]=node.index
                            tree_matrix_val[node.index][2]=int(node.left.index.split(':')[-1])

                        if not isinstance(node.right.index,str):
                            tree_matrix_val[node.index][1]=node.right.index
                        else:
                            tree_matrix_val[node.index][1]=node.index
                            tree_matrix_val[node.index][3]=int(node.right.index.split(":")[-1])
                    except:
                        pass

        self.max_route_len=0
        for u in sorted(self.result_dict.keys()):
            self.routes.append(self.get_route(self.result_dict[u]))
            self.max_route_len=max(len(self.routes[-1]),self.max_route_len)

        
        self.route_node_matrix_val = np.zeros((len(self.result_dict.keys()), self.max_route_len), dtype=np.int32)
        self.route_choice_matrix_val=np.zeros((len(self.result_dict.keys()),self.max_route_len),dtype=np.int32)
        self.mask_matrix_val=np.zeros((len(self.result_dict.keys()),self.max_route_len),dtype=np.int32)

        # Route matrix
        # mask-matrix
        for i,route in enumerate(self.routes):
            for a in range(self.max_route_len):
                try:
                    self.route_node_matrix_val[i][a]=route[a][0].index
                    self.route_choice_matrix_val[i][a]=route[a][1]
                    self.mask_matrix_val[i][a]=1.0
                except:
                    self.route_node_matrix_val[i][a]=0
                    self.route_choice_matrix_val[i][a]=0
                    self.mask_matrix_val[i][a]=0.0

        self.tree_matrix=theano.shared(value=tree_matrix_val,name='tree_matrix',borrow=True)
        self.route_node_matrix=theano.shared(value=self.route_choice_matrix_val,name=self.prefix+'route_node_matrix',borrow=True)
        self.route_choice_matrix=theano.shared(value=self.route_choice_matrix_val,name=self.prefix+'route_choice_matrix',borrow=True)
        self.mask_matrix=theano.shared(value=self.mask_matrix_val,name=self.prefix+'route_mask_matrix',borrow=True)

        # parameter_matrix_W
        wp_val=np.asarray(self.rng.uniform(low=-np.sqrt(6./(self.in_size)),
                                           high=np.sqrt(6./(self.in_size+2)),
                                           size=(len(self.nodes)+1,self.in_size)),dtype=theano.config.floatX)
        self.wp_matrix=theano.shared(value=wp_val,name="V_soft",borrow=True)
        self.params=[self.wp_matrix,]

    def build_graph(self):
        # 1
        nodes=self.route_node_matrix[self.y]
        choices=self.route_choice_matrix[self.y]
        mask=self.mask_matrix[self.y]
        # 2.
        wp=self.wp_matrix[nodes]
        # feature.dimshuffle(0,1,'x',2)
        node=T.sum(wp * self.x.dimshuffle(0,1,'x',2),axis=-1)

        #log_sigmoid=-T.mean(T.log(T.nnet.sigmoid(node*choices))*mask,axis=-1)
        log_sigmoid=T.mean(T.log(1+T.exp(-node*choices))*mask,axis=-1)

        cost=log_sigmoid*self.maskY   # matrix element-wise dot
        self.activation=cost.sum()/self.maskY.sum()



    def get_output(self, train=False):


        cls_idx = ins['cls_idx']
        word_bits_mask = ins['word_bitstr_mask']

        wp = self.wp_matrix[cls_idx]  # (n_s, n_t, n_n, d_l)
        # node_bias = self.b[cls_idx]                           # (n_s, n_t, n_n)

        # score = T.sum(features * node_embeds, axis=-1) + node_bias         # (n_s, n_t, n_n)
        node = T.sum( wp*x.dimshuffle(0, 1, 'x', 2), axis=-1)  # (n_s, n_t, n_n)
        prob_ = T.nnet.sigmoid(node * word_bits_mask)  # (n_s, n_t, n_n)
        prob = T.switch(T.eq(word_bits_mask, 0.0), 1.0, prob_)  # (n_s, n_t, n_n)
        log_prob = T.sum(T.log(self.eps + prob), axis=-1)  # (n_s, n_t)
        return log_prob



    def build_predict(self):
        n_steps=self.x.shape[0]
        batch_size=self.x.shape[1]

        fires=T.ones(shape=(n_steps,batch_size),dtype=np.int32)*self.tree[-1][0].index

        def predict_step(current_node,input_vector):
            # left nodes
            node_res_l=T.nnet.sigmoid(T.sum(self.wp_matrix[current_node]*input_vector,axis=-1))
            node_res_r=T.nnet.sigmoid(-1.0*T.sum(self.wp_matrix[current_node]*input_vector,axis=-1))

            choice=node_res_l>node_res_r
            next_node=self.tree_matrix[current_node,choice]
            labelings=self.tree_matrix[current_node,choice+2]

            return next_node,labelings,choice

        xresult,_=theano.scan(fn=predict_step,
                              outputs_info=[fires,None,None],
                              non_sequences=self.x,
                              n_steps=self.max_route_len)
        self.labels=xresult[1][-1]*self.maskY
        self.predict_labels=theano.function(inputs=[self.x,self.maskY],
                                           outputs=self.labels)
        #self.label_tool=theano.function([self.x],xresult)

    def get_prediction_function(self):
        return  self.predict_labels#,self.label_tool

    def get_route(self,node):
        route=[]
        parent=node.parent
        parent_choice=node.parent_choice
        route.append((parent,parent_choice))
        while(parent!=None):
            n_parent=parent.parent
            if n_parent!=None:
                parent_choice=parent.parent_choice
                route.append((n_parent,parent_choice))
            parent=parent.parent
        return route
