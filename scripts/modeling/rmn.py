#==================#
#=*= RMN Module =*=#
#==================#

# RMN Class for training Relationship Modeling Networks 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Embedding, Dense, Lambda, Input, Masking, Reshape
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.regularizers import Regularizer

from helper import pickle_object, load_pickled_object
from vector_math import find_nn_cos

# constants
MAX_SPAN_LENGTH = 50
NUM_TOPICS = 20
LAMBDA = 5.0

# hyperparameters
OPTIMIZER = 'adam'
BATCH_SIZE = 50
EPOCHS = 5

# saving tags
RMN_TAG = "rmn_%s"
MODEL = "model.h5"
ATTR = "attributes"

# attribute keys
N_TOP_KEY = 'num_topics'
LAMB_KEY  = 'lambda'
EMBED_KEY = 'emedding_matrix'
TOKEN_KEY = 'tokenizer_dict'
META_KEY  = 'metadata_dict'


class RMN(object):
    """
    Class for constructing a Relationship Modeling Network
    """
    
    def __init__(self):
        
        # model parameters
        self.num_topics = NUM_TOPICS
        self.lamb = LAMBDA
        
        # model attrbiutes
        self.embedding_matrix = None
        self.tokenizer_dict = None
        self.metadata_dict = None
        
        # models 
        self.model = None
        self.topic_model = None
        
    
    @property
    def embedding_dim(self):
        return self.embedding_matrix.shape[1]
    
    
    def model_loss(self):
        """Hinge loss function.
        """
        def custom_loss(y_true, y_pred):
            # hinge_loss
            y_true_normalized = K.l2_normalize(y_true, axis=-1)
            y_pred_normalized = K.l2_normalize(y_pred, axis=-1)
            dot_product = K.sum(y_true_normalized * y_pred_normalized, axis=-1)
            hinge_loss = K.mean(K.maximum(0., 1. - dot_product))

            return hinge_loss 

        return custom_loss
    
    
    def build_model(self):
        """Connstruct the RMN model architecture
        """
        # document span input
        vt = Input(shape=(self.embedding_dim, ), name='Span.Input')
    
        input_layers = [vt]
        embedding_layers = [vt]
        
        for col in self.metadata_dict.keys():
            
            input_layer = Input(shape=(1,), name= col + '.Input')
            
            # embedding layer for col
            embedding_init = Embedding(
                input_dim = self.metadata_dict[col]['input_dim'] + 1, 
                output_dim = self.embedding_dim,
                input_length = 1)(input_layer)
            
            # reshape
            embedding_layer = Reshape((self.embedding_dim, ), name=col + '.Embed.Layer')(embedding_init)
            
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
                   kernel_regularizer = Orthogonality(self.lamb),
                   name = "R")(dt)

        # compile
        model = tf.keras.Model(inputs=input_layers, outputs=rt)
        model.compile(optimizer = OPTIMIZER, loss = self.model_loss())

        self.model = model
    
    
    def build_topic_model(self, topic_layer = "Wd"):
        """Contruct model whose output is the topic distribution layer
        """
        topic_model = tf.keras.Model(
            inputs = self.model.input,
            outputs = self.model.get_layer(topic_layer).output)
        
        self.topic_model = topic_model
    
    def prep_y(self, y):
        """Returns the average of the vectors in each span of text
        """
        padded_spans = self.tokenizer_dict['tokenize_pad'](y)
        vector_spans = self.embedding_matrix[padded_spans].mean(axis=1)
        
        return vector_spans
    
    
    def prep_metadata(self, df):
        """Preps metadata for training or prediction
        """
        metadata_ids = [np.array(self.metadata_dict[col]['tokenize'](df[col]))
                        for col in self.metadata_dict.keys()]

        return metadata_ids
        
    
    def prep_inputs(self, df):
        """Preps metadata for training or prediction
        """
        vector_spans = self.prep_y(df['document'])
        metadata_ids = self.prep_metadata(df)
        inputs = [vector_spans] + metadata_ids
        
        return inputs
    
    
    def predict_topics(self, df):
        """Predicts the topic distributions for a df
        """
        
        # ensure the topic model has been built
        if self.topic_model is None:
            self.build_topic_model()
            
        topic_preds = self.topic_model.predict(x=self.prep_inputs(df))
        
        return topic_preds
    
    
    def fit(self, df, batch_size = BATCH_SIZE, epochs = EPOCHS):
        
        inputs = self.prep_inputs(df)
        y_true = self.prep_y(df['document'])
        
        self.model.fit(x = inputs, 
                       y = y_true, 
                       batch_size = batch_size, 
                       epochs = epochs)
    
    def save_rmn(self, name, save_path):
        """
        Save the model's weights, architecture and attributes
        """
        
        # assemble attribute dictionary
        attribute_dict = {
            N_TOP_KEY:  self.num_topics,
            LAMB_KEY:   self.lamb,
            EMBED_KEY:  self.embedding_matrix,
            TOKEN_KEY:  self.tokenizer_dict,
            META_KEY:   self.metadata_dict}
        
        # make directory for model
        model_path = os.path.join(save_path, RMN_TAG % name)
        os.mkdir(model_path)
        
        # save model weights
        self.model.save_weights(os.path.join(model_path, MODEL))
        
        # save model attributes
        pickle_object(attribute_dict, os.path.join(model_path, ATTR))
        
        
    def load_rmn(self, name, save_path):
        """
        Load the model, weights, architecture and attributes from a saved model
        """
        
        # make directory for model
        model_path = os.path.join(save_path, RMN_TAG % name)
        
        # load attributes
        attributes_dict = load_pickled_object(os.path.join(model_path, ATTR))
        
        # update attributes
        self.num_topics       = attributes_dict[N_TOP_KEY]
        self.lamb             = attributes_dict[LAMB_KEY]
        self.embedding_matrix = attributes_dict[EMBED_KEY]
        self.tokenizer_dict   = attributes_dict[TOKEN_KEY]
        self.metadata_dict    = attributes_dict[META_KEY]
        
        # construct identical model architecture
        self.build_model()
        
        # Load weights
        self.model.load_weights(os.path.join(model_path, MODEL))
        
    
    def inspect_topics(self, k_neighbors=10):
        """
        Ouput the nearest neighbors of every topic vector in
        the model's topic layer
        """
    
        # get embedding matrix, dim = [num_words, embedding_dim]
        E = self.embedding_matrix
        
        # get topic matrix, dim = [num_topics, embedding_dim]
        Wd = self.model.get_layer('Wd').get_weights()[0].T
        
        for i in range(Wd.shape[0]):
            
            neighbors, sim = find_nn_cos(Wd[i], E, k_neighbors)
            words = [self.tokenizer_dict['tokenizer'].index_word[v] for v in neighbors]
            
            print(20*"=" +"\n")
            print("Topic", i)
            print(words)
    
    
# Orthogonality Regularizer #

class Orthogonality(Regularizer):
    """Regularizer for discouraging non-orthogonal components.
    
    # Arguments
        lamb: Float; regularization penalty weight
    """

    def __init__(self, lamb = 1.):
        self.lamb = lamb

    def __call__(self, R):
        RRT = K.dot(R, K.transpose(R))
        I = K.eye(int(RRT.shape[0]))
        penalty = self.lamb * K.sqrt(K.sum(K.square(RRT - I)))
        
        return penalty