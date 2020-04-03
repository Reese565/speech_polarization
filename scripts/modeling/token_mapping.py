import os
import pandas as pd

<<<<<<< HEAD
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
=======
from functools import partial
>>>>>>> master

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# constants
MAX_TOKENS = 80


def ohe_attributes(subject_df):
    
    # extract speaker metadata attributes
    attributes = subject_df.columns.drop('speech')
    
    # set attributes to string
    subject_df['speakerid'] = subject_df['speakerid'].astype(str)
    subject_df['congress'] = subject_df['congress'].astype(str)

    # one-hot-encode speaker metadata
    for col in attributes:
        subject_df = pd.concat([subject_df, pd.get_dummies(subject_df[col])], axis = 1)

    return subject_df


def tokenize_pad(documents, tokenizer, max_span_len):
    """Tokenize and pad documents using a tokenizer
    """
    tokenized = tokenizer.texts_to_sequences(documents)
    padded = pad_sequences(tokenized, maxlen=max_span_len, padding = "post")
    
    return padded


def build_tokenizer_dict(document_df, max_span_len = MAX_TOKENS):
    """Returns a dictionary with useful properties of a tokenizer fit on document_df
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(document_df['document'])
    tokenizer_pad = partial(tokenize_pad, tokenizer=tokenizer, max_span_len=max_span_len)
    
    tokenizer_dict = {
        'tokenizer': tokenizer, 
        'tokenize_pad': tokenizer_pad, 
        'word_index': tokenizer.word_index,
        'max_span_length': max_span_len}
    
    return tokenizer_dict


def build_metadata_dict(document_df, metadata_columns):
    """Returns a dictionary with useful properties of tokenizers fit on metadata
    """
    metadata_dict = {}
    
    for col in metadata_columns:
        
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(document_df[col])
    
        metadata_dict[col] = {
            'tokenizer': tokenizer,
            'tokenize': tokenizer.texts_to_sequences,
            'token_index': tokenizer.word_index, 
            'input_dim': len(tokenizer.word_index)}        
        
    return metadata_dict


