import os

import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
import numpy as np


from tensorflow.keras.preprocessing.text import Tokenizer


os.chdir("../assembly/")
from constant import SPEECHES, SPEAKER_MAP, HB_PATH, EMBEDDINGS



# constants
GLOVE_DIMS = [50, 100, 200, 300]
EMBEDDING_DIM = GLOVE_DIMS[0]
GLOVE_PATH = os.path.join(EMBEDDINGS, 'glove6B/glove.6B.%dd.csv' % EMBEDDING_DIM)


# tokenizers = 'import from tokenizer file'

# speeches_word_index = tokenizers['speech']


def build_embedding_matrix(speeches_word_index, embedding_path = GLOVE_PATH):
    
    embeddings_index = pd.read_csv(GLOVE_PATH).to_dict(orient = 'list')

    embedding_matrix = np.zeros((len(speeches_word_index) + 1, EMBEDDING_DIM))
    for word, i in speeches_word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix



# def add_words_to_matrix(embedding_matrix):
    
#     return None
    
    
