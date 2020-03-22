<<<<<<< HEAD
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Embedding, Dense, Lambda, Input, Masking
=======
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
>>>>>>> master

from embeddings import EMBEDDING_DIM
from helper import pickle_object, load_pickled_object
from vector_math import find_nn_cos

# constants
MAX_SPAN_LENGTH = 50
NUM_TOPICS = 20

OPTIMIZER = 'adam'
BATCH_SIZE = 50
EPOCHS = 5

RMN_TAG = "rmn_%s"
MODEL = "model.h5"
ARCH = "architecture"
ATTR = "attributes"


class RMN(object):
    """
    Class for constructing a Relationship Modeling Network
    """
    
<<<<<<< HEAD
    def __init__(self, embedding_dim = EMBEDDING_DIM, num_topics = NUM_TOPICS):
        self.embedding_dim = embedding_dim
        self.num_topics = num_topics
        self.metadata_dict = None
        self.model = None
        self.loss = None
        self.inputs = None
        self.y_true = None
=======
    def __init__(self):
        
        # model attributes
        self.num_topics = NUM_TOPICS
        self.embedding_matrix = None
        self.tokenizer_dict = None
        self.metadata_dict = None
        
        # models 
        self.model = None
        self.topic_model = None
        
    
    @property
    def embedding_dim(self):
        return self.embedding_matrix.shape[1]
>>>>>>> master
    
    
    def model_loss(self, layer, lamb = 1.0):
        """Custom loss function to engourage 
        orthoganality of dictionary matrix R."""

        R = K.transpose(layer)
        
        def custom_loss(y_true, y_pred):

            hinge_loss = tf.keras.losses.hinge(y_true, y_pred)

            RR_t = K.dot(R, K.transpose(R))
            Id_mat = K.eye(self.embedding_dim)

            orth_penalty = K.sqrt(K.sum(K.square(RR_t - Id_mat)))

            return hinge_loss + lamb*orth_penalty

        return custom_loss
    
    def summary(self):
        """Standard summary function for keras model"""
        
        self.model.summary()
        
        return None
    
    def ingest_inputs(self, embeddings_matrix, train_data):
        """Feeds and assembles inputs to model"""
    
        # avergage span embeddings
        Vst_train = embeddings_matrix[train_data].mean(axis=1)

        inputs = [Vst_train]
        for key in self.metadata_dict.keys():
            inputs.append(self.metadata_dict[key]['input'])
            
        self.inputs = inputs
        self.y_true = Vst_train

        return None
    
    def build_model(self):
        """Connstruct the RMN model architecture
        """
        # document span input
        vt = Input(shape=(self.tokenizer_dict['max_span_length'], ), name='Span.Input')
    
<<<<<<< HEAD
    
    def build_model(self, metadata_dict):
        """"""
        
        self.metadata_dict = metadata_dict
        
        # input avg span embeddings
        vt = Input(shape=(self.embedding_dim,), name='Avg.Span.Embed.Input')

        ## initializing speaker metadata embeddings

=======
>>>>>>> master
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
                   # kernel_regularizer = Orthoganal(),
                   name = "R")(dt)

        # compile
        model = tf.keras.Model(inputs=input_layers, outputs=rt)
        self.loss = self.model_loss(rt)
        self.model = model

<<<<<<< HEAD
        return self
    
    
    def compile_RMN(self, optimizer = OPTIMIZER):
        
        self.model.compile(optimizer = optimizer, loss = self.loss)
        
        return None
    
    def fit(self, batch_size = 1, epochs = 1):
        
        inputs = self.inputs
        y_true = self.y_true
        
        self.model.fit(inputs, y_true, batch_size = batch_size, epochs = epochs)
        
        return None
        
        
=======
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
            'num_topics': self.num_topics,
            'emedding_matrix': self.embedding_matrix,
            'tokenizer_dict': self.tokenizer_dict,
            'metadata_dict': self.metadata_dict}
        
        # make directory for model
        model_path = os.path.join(save_path, RMN_TAG % name)
        os.mkdir(model_path)
        
        # save model weights
        self.model.save(os.path.join(model_path, MODEL))
        
        # save model architecture
        pickle_object(self.model.to_json(), os.path.join(model_path, ARCH))
        
        # save model attributes
        pickle_object(attribute_dict, os.path.join(model_path, ATTR))
        
        
    def load_rmn(self, name, save_path):
        """
        Load the model, weights, architecture and attributes from a saved model
        """
        
        # make directory for model
        model_path = os.path.join(save_path, RMN_TAG % name)
        
        # Load architecture and weights
        self.model = model_from_json(load_pickled_object(os.path.join(model_path, ARCH)))
        self.model.load_weights(os.path.join(model_path, MODEL))
        
        # load attributes
        attributes_dict = load_pickled_object(os.path.join(model_path, ATTR))
        
        # update attributes
        self.num_topics = attributes_dict['num_topics']
        self.embedding_matrix = attributes_dict['emedding_matrix']
        self.tokenizer_dict = attributes_dict['tokenizer_dict']
        self.metadata_dict = attributes_dict['metadata_dict']
       
    
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
            
>>>>>>> master
