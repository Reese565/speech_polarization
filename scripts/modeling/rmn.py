
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
    
    
    def model_loss(self, layer, lamb = 1.0):

        R = K.transpose(layer)

        def custom_loss(y_true, y_pred):

            hinge_loss = tf.keras.losses.hinge(y_true, y_pred)

            RR_t = K.dot(R, K.transpose(R))
            Id_mat = K.eye(self.embedding_dim)

            orth_penalty = K.sqrt(K.sum(K.square(RR_t - Id_mat)))

            return hinge_loss + lamb*orth_penalty

        return custom_loss
    
    
    
    
    def build_model(self, metadata_dict):
        
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
        model.compile(optimizer = OPTIMIZER, loss = self.model_loss(rt))

        return model
    
    
    