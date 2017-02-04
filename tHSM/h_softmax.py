import numpy as np
import theano
import theano.tensor as T
import Queue

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



class HuffmanNode(object):
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
            if isinstance(self.left[1],HuffmanNode):
                self.left[1].preorder(polarity+[-1],param+[self.index],collector)
            else:
                #collector.append((self.left[1],self.left[0],path+[-1]))
                collector.append((self.left[1],param+[self.index], polarity + [-1]))
        if self.right:
            if isinstance(self.right[1],HuffmanNode):
                self.right[1].preorder(polarity+[1],param+[self.index],collector)
            else:
                #collector.append((self.right[1],self.right[0],path+[1]))
                collector.append((self.right[1],param+[self.index], polarity + [1]))
        return collector


def build_huffman(frequenties):
    Q=Queue.PriorityQueue()
    for v in frequenties:  #((freq,word),index)
        Q.put(v)
    idx=0
    while Q.qsize()>1:
        l,r=Q.get(),Q.get()
        node=HuffmanNode(l,r,idx)
        idx+=1
        freq=l[0]+r[0]
        Q.put((freq,node))
    return Q.get()[1]


def load_prefix(rel_freq,meta_file=None):
    freq = zip(rel_freq, range(len(rel_freq)))
    tree = build_huffman(freq)
    x = tree.preorder()
    x = sorted(x, key=lambda z: z[0])


if __name__ == '__main__':

    freq = [8.167, 1.492, 2.782, 4.253, 12.702, 2.228, 2.015, 6.094, 6.966, 0.153, 0.747, 4.025, 2.406, 6.749, 7.507,
            1.929, 0.095, 5.987, 6.327, 9.056, 2.758, 1.037, 2.365, 0.150, 1.974, 0.074]
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
def build_binary_tree(frequenties):
    Q=Queue.PriorityQueue()

    # fqs: frequenties, widx: word index
    for fqs,widx in frequenties:
        Q.put(TNode(index=widx,freq=fqs))
    count=0
    while Q.qsize() > 1:
        l,r=Q.get(),Q.get()
        node=TNode(index=count,left=l,right=r,freq=l.freq+r.freq)
        count+=1
        l.parent=node
        l.parent_choice=-1
        r.parent=node
        r.parent_choice=1
        Q.put(node)
    return Q.get()
    '''
        pairs=[]
        if len(current_layer) > 1:
            while(len(current_layer)>1):
                pairs.append(current_layer[:2])
                current_layer=current_layer[2:]
        else:
            pairs=[current_layer]
            current_layer=[]
        new_layer=[]
        for p in pairs:
            node=TreeNode(index=count,left=p[0],right=p[1])
            count+=1
            p[0].parent=node
            p[0].parent_choice=-1
            p[1].parent=node
            p[1].parent_choice=1
            new_layer.append(node)
        if len(current_layer)>0:
            new_layer.extend(current_layer)
            current_layer=[]
        layers.append(new_layer)
        current_layer=new_layer

    return layers
    '''