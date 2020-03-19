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



def build_tokenizer_dict(document_df, feature_columns, max_span_len = MIN_TOKENS):

    tokenizers = {}
    for col in ['document'] + feature_columns:
    
        # create tokenizer for documents or meta datum
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(document_df[col])
        tokenized = tokenizer.texts_to_sequences(document_df[col])
        
        if col == 'document':
            tokenized = pad_sequences(tokenized, maxlen=max_span_len, padding="post")
            
        tokenizers[col] = {
            'tokenizer': tokenizer,
            'tokenized': tokenized,
            'token_index': tokenizer.word_index}        
        
    return tokenizers



def build_metadata_dict(documents_df, feature_columns):
    
    metadata_dict = {col:{'input_dim': documents_df[col].unique().shape[0]} 
                     for col in feature_columns}
        
    return metadata_dict


