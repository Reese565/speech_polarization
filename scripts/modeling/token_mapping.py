import os
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.chdir('../assembly/')
from document import WINDOW


def ohe_attribures(subject_df):
    
    # extract speaker metadata attributes
    attributes = subject_df.columns.drop('speech')
    
    # set attributes to string
    subject_df['speakerid'] = subject_df['speakerid'].astype(str)
    subject_df['congress'] = subject_df['congress'].astype(str)

    # one-hot-encode speaker metadata
    for col in attributes:
        subject_df = pd.concat([subject_df,pd.get_dummies(subject_df[col])], axis = 1)


    return subject_df



def build_tokenizer_dict(subject_df):
    
    max_len = WINDOW + 1
    
    # building tokenizers, word indecies, and train data
    speech_tokenizer = Tokenizer()
    speech_tokenizer.fit_on_texts(subject_df['speech'].values)
    speeches_word_index = speech_tokenizer.word_index
    speeches_train = speech_tokenizer.texts_to_sequences(subject_df['speech'].values)
    speeches_train_padded = pad_sequences(speeches_train, maxlen=WINDOW + 1, padding="post")
    


    tokenizers = {}
    tokenizers['speech'] = {'tokenizer': speech_tokenizer,
                            'train': speeches_train,
                            'train_padded': speeches_train_padded,
                            'word_index': speeches_word_index}

#     for col in attributes:
#         tokenizer = Tokenizer()
#         try:
#             tokenizer.fit_on_texts(subject_df[col].values)
#         except:
#             print(col)
#             raise
#         tokenizers[col] = {}
#         tokenizers[col]['train'] = tokenizer.texts_to_sequences(subject_df[col].values)
#         tokenizers[col]['word_index'] = tokenizer.word_index
#         tokenizers[col]['tokenizer'] = tokenizer
        
        
    return tokenizers


def build_metadata_dict(feature_columns, subject_df):
    
    # one-hot-encoded speaker metadata inputs

    metadata_dict = {}

    for col in feature_columns:
        df = subject_df[subject_df[col].unique()].values
        dim = df.shape[1]
        metadata_dict[col] = {'input': df, 'input_dim': dim}
        
    return metadata_dict


