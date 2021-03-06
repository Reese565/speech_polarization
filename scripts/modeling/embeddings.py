import os
import pickle

import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences


os.chdir("../assembly/")
from constant import EMBEDDINGS


# constants
GLOVE_DIMS = [50, 100, 200, 300]
EMBEDDING_DIM = GLOVE_DIMS[0]
GLOVE_PATH = os.path.join(EMBEDDINGS, 'glove6B/glove.6B.%dd.csv' % EMBEDDING_DIM)


def fetch_embeddings(embeddings_dim = EMBEDDING_DIM):
    
    path = os.path.join(EMBEDDINGS, 'glove6B/glove.6B.%dd.csv' % EMBEDDING_DIM)
    
    embeddings_index = pd.read_csv(path).to_dict(orient = 'list')
    
    return embeddings_index



def build_embedding_matrix(word_index, embeddings_index, stopwords=[]):
    
    # get the embedding dimension
    embedding_dim = len(embeddings_index['the'])
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and word not in stopwords:
            # words not found in embedding index and stopwords will be all-zeros.
            embedding_matrix[i] = embedding_vector 
            
    embedding_matrix = embedding_matrix.astype('float16')
    
    return embedding_matrix