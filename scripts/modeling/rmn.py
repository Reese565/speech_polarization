import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Embedding, Dense, Lambda, Input, Reshape
from embeddings import EMBEDDING_DIM


# constants
MAX_SPAN_LENGTH = 50
NUM_TOPICS = 20
OPTIMIZER = 'adam'


class RMN(object):
    
    def __init__(self):
        
        self.num_topics = NUM_TOPICS
        
        self.embedding_matrix = None
        self.tokenizer_dict = None
        self.metadata_dict = None
        
        self.model = None
        self.topic_model = None
        
    
    @property
    def embedding_dim(self):
        return self.embedding_matrix.shape[1]
    
    
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
    
    
    def build_model(self):
        """Connstruct the RMN model architecture
        """
        # document span input
        vt = Input(shape=(self.tokenizer_dict['max_span_length'], ), name='Span.Input')
    
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
        model.compile(optimizer = OPTIMIZER, loss = self.model_loss(rt))

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
                        for col in metadata_dict.keys()]

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
    
    
    # def save_rmn(self, name, save_path):
        
        
