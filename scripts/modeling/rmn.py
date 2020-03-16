import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Embedding, Dense, Lambda, Input, Masking

from embeddings import EMBEDDING_DIM


# constants
NUM_TOPICS = 20
OPTIMIZER = 'adam'


class RMN(object):
    
    def __init__(self, embedding_dim = EMBEDDING_DIM, num_topics = NUM_TOPICS):
        self.embedding_dim = embedding_dim
        self.num_topics = num_topics
        self.metadata_dict = None
        self.model = None
        self.loss = None
        self.inputs = None
        self.y_true = None
    
    
    def model_loss(self, layer, lamb = 1.0):

        R = K.transpose(layer)

        def custom_loss(y_true, y_pred):

            hinge_loss = tf.keras.losses.hinge(y_true, y_pred)

            RR_t = K.dot(R, K.transpose(R))
            Id_mat = K.eye(self.embedding_dim)

            orth_penalty = K.sqrt(K.sum(K.square(RR_t - Id_mat)))

            return hinge_loss + lamb*orth_penalty

        return custom_loss
    
    def summary(self):
        
        self.model.summary()
        
        return None
    
    def ingest_inputs(self, embeddings_matrix, train_data):
    
        # avergage span embeddings
        
#         speeches_train_padded = self.metadata_dict['speech']['train_padded']
        Vst_train = embeddings_matrix[train_data].mean(axis=1)

        inputs = [Vst_train]
        for key in self.metadata_dict.keys():
            inputs.append(self.metadata_dict[key]['input'])
            
        self.inputs = inputs
        self.y_true = Vst_train

        return None
    
    
    
    def build_model(self, metadata_dict):
        
        self.metadata_dict = metadata_dict
        
        # input avg span embeddings
        vt = Input(shape=(self.embedding_dim,), name='Avg.Span.Embed.Input')

        ## initializing speaker metadata embeddings

        input_layers = [vt]
        embedding_layers = [vt]
        for col in metadata_dict.keys():

            # one-hot-encoded embeedings layers
            input_dim = metadata_dict[col]['input_dim']
            input_layer = Input(shape=(input_dim,), name= col + '.Embed.Input')
            embedding_init = (Dense(units = self.embedding_dim,
                                    kernel_initializer = 'glorot_normal',
                                    input_shape = (input_dim, ),
                                    activation = "linear",
                                    name = 'C_' + col)(input_layer))


            input_layers.append(input_layer)
            embedding_layers.append(embedding_init)

        # concat speaker metadata embeddings
        _ht = tf.keras.layers.Concatenate(axis=1, name = 'Concat.Layer')(embedding_layers)

        # dense layer
        ht = Dense(units = self.embedding_dim, input_shape = (_ht.shape[1], ), activation = "relu", name = "Wh")(_ht)

        # dense layer with softmax activation, (where previous states will eventually be inserted) 
        dt = Dense(units = self.num_topics, input_shape = (self.embedding_dim, ), activation = "softmax", name = "Wd")(ht)

        # reconstruction layer
        rt = Dense(units = self.embedding_dim,
                   input_shape = (self.num_topics, ),
                   activation = "linear",
#                    kernel_regularizer = Orthoganal(),
                   name = "R")(dt)

        model = tf.keras.Model(inputs=input_layers, outputs=rt)
        self.loss = self.model_loss(rt)
        self.model = model

        return self
    
    
    def compile_RMN(self, optimizer = OPTIMIZER):
        
        self.model.compile(optimizer = optimizer, loss = self.loss)
        
        return None
    
    def fit(self, batch_size = 1, epochs = 1):
        
        inputs = self.inputs
        y_true = self.y_true
        
        self.model.fit(inputs, y_true, batch_size = batch_size, epochs = epochs)
        
        return None
        
        