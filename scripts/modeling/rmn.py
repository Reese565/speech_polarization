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
from tensorflow.keras.layers import Embedding, Dense, Lambda, Input, Masking, Reshape, Concatenate
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.optimizers import Adam

from rmn_data_generator import RMN_DataGenerator
from helper import pickle_object, load_pickled_object
from vector_math import find_nn_cos

# constants
MAX_SPAN_LENGTH = 50
NUM_TOPICS = 20
LAMBDA = 1.0
GAMMA = 1.0

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
DIM_KEY = 'meta_embedding_dim'


class RMN(object):
    """
    Class for constructing a Relationship Modeling Network
    """
    
    def __init__(self):
        
        # model attrbiutes
        self.num_topics = NUM_TOPICS
        self.embedding_matrix = None
        self.meta_embedding_dim = None
        self.tokenizer_dict = None
        self.metadata_dict = None
        
        # inference attributes
        self.infer_embedding_matrix = None
        self.infer_tokenizer_dict = None
        
        # models 
        self.model = None
        self.topic_model = None
        
    
    @property
    def embedding_dim(self):
        return self.embedding_matrix.shape[1]
    
    @property
    def topic_matrix(self):
        """Return the topic matrix associated with the rmn"""
        # dim = [num_topics, embedding_dim]
        return self.model.get_layer('Wd').get_weights()[0].T
    
    @property
    def tuned_embedding_matrix(self):
        """Return the current embedding matrix of the rmn"""
        return rmn.model.get_layer('Span.Embedding').get_weights()[0]
    
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
    
    
    def build_model(self, embedding_trainable=False, bias_reconstruct=True,
                    gamma = 1., theta = 1., omega = 1., word_dropout = 0.5):
        """Connstruct the RMN model architecture
        """
        # Span Input
        span_input = Input(shape=(self.tokenizer_dict['max_span_length'],), 
                           name='Span.Input')
        span_embedding = Embedding(input_dim=len(self.tokenizer_dict['word_index']) + 1, 
                                   output_dim=self.embedding_dim, 
                                   weights=[self.embedding_matrix],
                                   input_length=self.tokenizer_dict['max_span_length'],
                                   trainable=embedding_trainable, 
                                   name = 'Span.Embedding')(span_input)
        
        # Mask for randomly dropping words
        dropout_mask = K.stack(
            [K.random_binomial((span_embedding.shape[1],), p=word_dropout)]*span_embedding.shape[2], axis=1)
        # Average over the remaining words
        span_avg = Lambda(lambda x: K.mean(x * K.expand_dims(dropout_mask, axis=0), axis=1), 
                          name = "Span.Avg.Layer")(span_embedding)

        input_layers = [span_input]
        embedding_layers = [span_avg]
        
        for col in self.metadata_dict.keys():
            input_layer = Input(shape=(1,), name= col + '.Input')
            
            # embedding layer for col
            embedding_init = Embedding(
                input_dim = self.metadata_dict[col]['input_dim'] + 1, 
                output_dim = self.meta_embedding_dim,
                input_length = 1)(input_layer)
            
            # reshape
            embedding_layer = Reshape((self.meta_embedding_dim, ), name=col + '.Embed.Layer')(embedding_init)
            
            input_layers.append(input_layer)
            embedding_layers.append(embedding_layer)

        # concatenate span vector with metadata embeddings
        _ht = Concatenate(axis=1, name = 'Concat.Layer')(embedding_layers)

        # dense layer
        ht = Dense(units = self.embedding_dim, 
                   input_shape = (_ht.shape[1], ), 
                   activation = "relu", name = "Wh")(_ht)

        # dense layer whose output is a probability distribution
        dt = Dense(units = self.num_topics, 
                   input_shape = (self.embedding_dim, ), 
                   activation = "softmax",
                   activity_regularizer = Purity(gamma, theta, omega),
                   name = "Wd")(ht)

        # reconstruction layer
        rt = Dense(units = self.embedding_dim,
                   input_shape = (self.num_topics, ),
                   activation = "linear",
                   use_bias = bias_reconstruct,
                   kernel_regularizer = Orthogonality(self.lamb),
                   name = "R")(dt)

        # compile
        model = tf.keras.Model(inputs=input_layers, outputs=rt)
        #model.compile(optimizer = OPTIMIZER, loss='mean_squared_error')
        model.compile(optimizer = OPTIMIZER, loss = self.model_loss())
        self.model = model
        
        # build associated topic model
        self.build_topic_model()
        
    
    def set_topic_vectors(self, words):
        """Set the topic vectors with vectors corresponding to the given words
        """
        # get the word ids
        word_ids = self.tokenizer_dict['tokenize_pad'](words)[:,0]
        
        # replicate associated weights up to num_topics
        weights = np.tile(self.embedding_matrix[word_ids], 
                          (-(self.num_topics // -len(words)),1))[:self.num_topics]
        
        # set weights layer weights
        r = self.model.get_layer("R")
        if len(r.get_weights()) == 1:
            r.set_weights([weights])
        else:
            r.set_weights([weights, r.get_weights()[1]])
        
        
    def build_topic_model(self, topic_layer = "Wd"):
        """Contruct model whose output is the topic distribution layer
        """
        topic_model = tf.keras.Model(
            inputs = self.model.input,
            outputs = self.model.get_layer(topic_layer).output)
        
        self.topic_model = topic_model
          
    
    def prep_spans(self, documents):
        """Returns the lists of word ids associated with the text
        """
        return self.tokenizer_dict['tokenize_pad'](documents)
    
    
    def prep_metadata(self, df):
        """Preps metadata for training or prediction
        """
        metadata_x = [np.array(self.metadata_dict[col]['tokenize'](df[col]))
                      for col in self.metadata_dict.keys()]

        return metadata_x
        
    
    def prep_X(self, df, for_training=False):
        """Preps metadata and spans for training or prediction
        """
        spans_y = self.prep_spans(df['document'])
        metadata_x = self.prep_metadata(df)
        X = [spans_y] + metadata_x
        
        if for_training:
            y = self.embedding_matrix[spans_y].mean(axis=1)
            return X, y
        else:
            return X

    def predict_y(self, df, use_generator=True):
        """Predicts the rmn outputs for a df
        """
        # ensure the topic model has been built
        if self.topic_model is None:
            self.build_topic_model()
        
        if use_generator:
            return self.predict_with_generator(df, self.model)
        else:
            return self.predict_(df, self.model)
    
    
    def predict_topics(self, df, use_generator=True):
        """Predicts the topic distributions for a df
        """        
        # ensure the topic model has been built
        if self.topic_model is None:
            self.build_topic_model()
        
        if use_generator:
            return self.predict_with_generator(df, self.topic_model)
        else:
            return self.predict_(df, self.topic_model)

        
    def predict_(self, df, model):
        """Makes a predictions for a df with a model
        """
        return model.predict(x=self.prep_X(df))
        
    
    def predict_with_generator(self, df, model):
        """Predict topic distributions with a generator
        """
        # Make sure data is not empty
        assert not df.empty

        # Calculate good batch size, 
        batch_size = max(1, min(10000, df.shape[0] // 10))
        n_batches = df.shape[0] // batch_size

        if n_batches < 2: 
            return self.predict_(df, model)
        else:
            # calculate remainder batch size
            r = df.shape[0] % batch_size
            if r == 0:
                g_index = df.index[:-batch_size]
                r_index = df.index[-batch_size:]
            else:
                g_index = df.index[:-r]
                r_index = df.index[-r:]

            # Make generator
            g = RMN_DataGenerator(self, df.loc[g_index], batch_size=batch_size, shuffle=False)

            # Predict on remainder batch
            r_pred = self.predict_(df.loc[r_index], model)
            # predict on generated batches
            g_pred = model.predict_generator(g, use_multiprocessing=True, workers=10, verbose=1)

            assert r_pred.shape[1] == g_pred.shape[1]
            pred = np.vstack([g_pred, r_pred])

            return pred
        
    
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
            META_KEY:   self.metadata_dict, 
            DIM_KEY:    self.meta_embedding_dim}
        
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
        self.num_topics         = attributes_dict[N_TOP_KEY]
        self.lamb               = attributes_dict[LAMB_KEY]
        self.embedding_matrix   = attributes_dict[EMBED_KEY]
        self.tokenizer_dict     = attributes_dict[TOKEN_KEY]
        self.metadata_dict      = attributes_dict[META_KEY]
        self.meta_embedding_dim = attributes_dict[DIM_KEY] 
        
        # construct identical model architecture
        self.build_model()
        
        # Load weights
        self.model.load_weights(os.path.join(model_path, MODEL))
        
        # build associated topic model
        self.build_topic_model()
        
    
    def inspect_topics(self, which_topics='all', k_neighbors=10):
        """
        Ouput the nearest neighbors of every topic vector in
        the model's topic layer
        """
        if which_topics == 'all':
            which_topics = range(self.num_topics) 
        
        if (self.infer_embedding_matrix is None or 
            self.infer_tokenizer_dict is None):
            self.infer_embedding_matrix = self.embedding_matrix
            self.infer_tokenizer_dict = self.tokenizer_dict
        
        E = self.infer_embedding_matrix # dim = [vocab_size, embedding_dim]
        Wd = self.topic_matrix          # dim = [num_topics, embedding_dim]
        
        for i in which_topics:
            # find nearest neighbors to topic
            neighbors, sim = find_nn_cos(Wd[i], E, k_neighbors)
            words = [self.infer_tokenizer_dict['tokenizer'].index_word[v] for v in neighbors]
            print(20*"=" +"\n")
            print("Topic", i)
            print(words)
    
    
# Orthogonality Regularizer #

class Orthogonality(Regularizer):
    """
    Regularizer for penalizing non-orthogonal components of a weight matrix.
    
    Args:
    - lamb: (Float) regularization penalty weight
    """

    def __init__(self, lamb = 1.):
        self.lamb = lamb

    def __call__(self, R):
        """Returns a component dependence penalty for matrix R
        """
        RRT = K.dot(R, K.transpose(R))
        I = K.eye(RRT.shape.as_list()[0])
        penalty = self.lamb * K.sqrt(K.sum(K.square(RRT - I)))
        
        return penalty
    
    
# Topic Purity Regularizer #

class Purity(Regularizer):
    """Regularizer for penalizing highly impure probability distributions
    """
    def __init__(self, gamma = 1., theta = 1., omega = 1.):
        self.gamma = gamma
        self.theta = theta
        self.omega = omega

    def __call__(self, p):
        """Returns the avergage shannon entropy of the distribution(s) p
        """
        # calculate impurity and concentration
        impurity = K.sum(p*-K.log(p)/K.log(K.constant(2)), axis=-1)
        concentration = K.max(p, axis=-1)
        # calculate batch similarity
        ppt = K.dot(p, K.transpose(p)) 
        similarity = K.mean(ppt) - K.mean(tf.linalg.diag_part(ppt))
        
        penalty = (self.gamma * K.mean(impurity) + 
                   self.theta * K.mean(concentration) + 
                   self.omega * similarity)
        
        return penalty
    