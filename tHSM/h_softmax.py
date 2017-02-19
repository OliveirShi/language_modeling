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


class H_Softmax(object):

    def __init__(self,shape,x,y_node,y_choice,y_bit_mask,maskY):
        self.prefix='h_softmax_'

        self.in_size,self.out_size=shape
        # in_size:size,mb_size=out_size
        self.x=x
        self.y_node=y_node
        self.y_choice=y_choice
        self.y_bit_mask=y_bit_mask
        self.maskY=maskY
        self.rng=np.random.RandomState(12345)
        wp_val = np.asarray(self.rng.uniform(low=-np.sqrt(6. / (self.in_size)),
                                             high=np.sqrt(6. / (self.in_size + 2)),
                                             size=(self.out_size, self.in_size)), dtype=theano.config.floatX)
        self.wp_matrix = theano.shared(value=wp_val, name="V_soft", borrow=True)

        self.params = [self.wp_matrix, ]
        self.build_graph()
        #self.build_predict()



    def build_graph(self):
        wp=self.wp_matrix[self.y_node]
        # feature.dimshuffle(0,1,'x',2)
        node=T.sum(wp * self.x.dimshuffle(0,1,'x',2),axis=-1)

        #log_sigmoid=-T.mean(T.log(T.nnet.sigmoid(node*choices))*mask,axis=-1)
        log_sigmoid=T.mean(T.log(1+T.exp(-node*self.y_choice))*self.y_bit_mask,axis=-1)

        cost=log_sigmoid*self.maskY   # matrix element-wise dot
        self.activation=cost.sum()/self.maskY.sum()



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
        return  self.predict_labels

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
