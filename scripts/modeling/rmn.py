import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Embedding, Dense, Lambda, Input, Reshape
from embeddings import EMBEDDING_DIM


# constants
MAX_SPAN_LENGTH = 50
NUM_TOPICS = 20
OPTIMIZER = 'adam'

class RMN(object):
    
    def __init__(
        self, 
        embedding_dim = EMBEDDING_DIM, 
        num_topics = NUM_TOPICS):
        
        self.embedding_dim = embedding_dim
        self.num_topics = num_topics
        self.model = None
    
    def model_loss(self, rt_layer, lamb = 1.0):
        R = K.transpose(rt_layer)
        
        def custom_loss(y_true, y_pred):
            hinge_loss = tf.keras.losses.hinge(y_true, y_pred)
            
            RR_t = K.dot(R, K.transpose(R))
            Id_mat = K.eye(self.embedding_dim)
            orth_penalty = K.sqrt(K.sum(K.square(RR_t - Id_mat))) # axis = 1?
            
            return hinge_loss + lamb*orth_penalty
        
        return custom_loss
    
    def build_model(self, metadata_dict):

        vt = Input(shape=(MAX_SPAN_LENGTH,), name='Span.Input')
    
        input_layers = [vt]
        embedding_layers = [vt]
        for col in metadata_dict.keys():
            # embedding layers
            input_layer = Input(shape=(1,), name= col + '.Input')
            embedding_init = Embedding(
                input_dim = metadata_dict[col]['input_dim']+1, 
                output_dim = self.embedding_dim,
                input_length = 1)(input_layer)
            
            # reshape
            embedding_layer = Reshape((self.embedding_dim,), name=col + '.Embed.Layer')(embedding_init)
            
            input_layers.append(input_layer)
            embedding_layers.append(embedding_layer)

        # concat speaker metadata embeddings
        _ht = tf.keras.layers.Concatenate(axis=1, name = 'Concat.Layer')(embedding_layers)

        # dense layer
        ht = Dense(units = self.embedding_dim, 
                   input_shape = (_ht.shape[1], ), 
                   activation = "relu", name = "Wh")(_ht)

        # dense layer with softmax activation, (where previous states will eventually be inserted) 
        dt = Dense(units = self.num_topics, 
                   input_shape = (self.embedding_dim, ), 
                   activation = "softmax", name = "Wd")(ht)

        # reconstruction layer
        rt = Dense(units = self.embedding_dim,
                   input_shape = (self.num_topics, ),
                   activation = "linear",
                   # kernel_regularizer = Orthoganal(),
                   name = "R")(dt)

        model = tf.keras.Model(inputs=input_layers, outputs=rt)
        
        model.compile(optimizer = OPTIMIZER, loss = self.model_loss(rt))
        # model.compile(optimizer = OPTIMIZER, loss='mean_squared_error')

        self.model = model