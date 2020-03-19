import os
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.chdir('../assembly/')
from document import MIN_TOKENS


def ohe_attributes(subject_df):
    
    # extract speaker metadata attributes
    attributes = subject_df.columns.drop('speech')
    
    # set attributes to string
    subject_df['speakerid'] = subject_df['speakerid'].astype(str)
    subject_df['congress'] = subject_df['congress'].astype(str)

    # one-hot-encode speaker metadata
    for col in attributes:
        subject_df = pd.concat([subject_df,pd.get_dummies(subject_df[col])], axis = 1)

    return subject_df



def build_tokenizer_dict(document_df, max_span_len = MIN_TOKENS):

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(document_df['document'])
    
    def tokenizer_padder(documents):
        tokenized = tokenizer.texts_to_sequences(documents)
        padded = pad_sequences(tokenized, maxlen=max_span_len, padding = "post")
        return padded
    
    tokenizer_dict = {
        'tokenizer': tokenizer, 
        'tokenizer_padder': tokenizer_padder, 
        'word_index': tokenizer.word_index}
    
    return tokenizer_dict



def build_metadata_dict(document_df, metadata_columns):

    metadata_dict = {}
    
    for col in metadata_columns:
        
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(document_df[col])
    
        metadata_dict[col] = {
            'tokenizer': tokenizer,
            'token_index': tokenizer.word_index, 
            'input_dim': len(tokenizer.word_index)}        
        
    return metadata_dict

