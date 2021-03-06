import os
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from functools import partial


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# constants
MAX_TOKENS = 20


def tokenize_pad(documents, tokenizer, max_span_len, padding="post"):
    """Tokenize and pad documents using a tokenizer
    """
    tokenized = tokenizer.texts_to_sequences(documents)
    padded = pad_sequences(tokenized, maxlen=max_span_len, padding=padding)
    
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
        
        tokenizer = Tokenizer(filters='')
        tokenizer.fit_on_texts(document_df[col])
    
        metadata_dict[col] = {
            'tokenizer': tokenizer,
            'tokenize': tokenizer.texts_to_sequences,
            'token_index': tokenizer.word_index, 
            'input_dim': len(tokenizer.word_index)}        
        
    return metadata_dict
