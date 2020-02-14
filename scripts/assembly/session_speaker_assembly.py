import pandas as pd
import numpy as np
import os
from multiprocessing import  Pool
from itertools import product


# google storage bucket data paths
DATA_PATH = "gs://rwc1/data/"
HB_PATH = os.path.join(DATA_PATH, "hein-bound/")

# data file type
BY_SPEAKER = "byspeaker_2gram_%s.txt"
SPEAKER_MAP = "%s_SpeakerMap.txt"


# import master vocaulary with phrase classifications
phrases_classes = pd.read_csv(os.path.join(DATA_PATH, "vocabulary/master_list.txt"), sep = "|")



# helper function to parallelize operations
def parallelize_dataframe(df, func, args, n_cores=2):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df



def session_phrases(session):
    """Session phrase classes and counts by speakerid"""
    
    # import speaker bigrams for session X
    speaker_phrases = pd.read_csv(os.path.join(HB_PATH, BY_SPEAKER % session), sep = "|")

    # import speaker map for session X
    speaker_map_df = pd.read_csv(os.path.join(HB_PATH, SPEAKER_MAP % session), sep = "|")
    
    # generating valid session bigrams
    session_phrase_df = (speaker_phrases
                   .merge(phrases_classes, how = "inner", on ="phrase")
                  )
    
    return speaker_map_df, session_phrase_df



def select_phrase_classes(session_phrase_df, classes, ngram = 'bigram'):
    """"""
    # valid session bigrams
    valid_phrase_df = session_phrase_df[session_phrase_df['_classify'].isin(classes)]
    
    if ngram == 'unigram':
        # valid session unigram conversion
        valid_phrase_df['phrase'] =  valid_phrase_df['phrase'].str.split(' ')

    return valid_phrase_df



def mentions(df):
    """Takes in a dataframe with phrase 
        and occurance count columns and returns flat list 
        with multiples of the bigram phrases by occurrence"""
    
    def multiply_phrases(row):
        """Repeats phrase for each occurrence and
        returns a flat list"""
        
        # repeat phrase for each occurrence
        multi_phrase = [row['phrase']]*row['count']
        
        # flatten to an array
        flat_array = np.ravel(multi_phrase)
        
        # coerce to list object
        flat_list = list(flat_array)
        
        return flat_list
    
    # apply multi_phrase to each row in df
    df['phrase'] = df.apply(lambda row: multiply_phrases(row), axis=1)
        
    return df



def speaker_docs(valid_phrase_df):
    """"""
    phrase_counts = parallelize_dataframe(valid_phrase_df, mentions)
    speaker_phrases = phrase_counts.groupby('speakerid')['phrase'].sum()
    
    return speaker_phrases