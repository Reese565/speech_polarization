import pandas as pd
import numpy as np
import os
import dask.dataframe as dd

# google storage bucket data paths
DATA_PATH = "gs://rwc1/data/"
HB_PATH = os.path.join(DATA_PATH, "hein-bound/")

# data file type
BY_SPEAKER = "byspeaker_2gram_%s.txt"
SPEAKER_MAP = "%s_SpeakerMap.txt"


# import master vocaulary with phrase classifications
phrase_classes_dd = dd.read_csv(os.path.join(DATA_PATH, "vocabulary/master_list.txt"), sep = "|")


def session_phrases_dd(session):
    """Session phrase classes and counts by speakerid"""
    
    # import speaker bigrams for session X
    speaker_phrases = dd.read_csv(os.path.join(HB_PATH, BY_SPEAKER % session), sep = "|")

    # import speaker map for session X
    speaker_map_dd = dd.read_csv(os.path.join(HB_PATH, SPEAKER_MAP % session), sep = "|")
    
    # generating valid session bigrams
    session_phrase_dd = (speaker_phrases.merge(phrase_classes_dd, how = "inner", on ="phrase"))
    
    return speaker_map_dd, session_phrase_dd


def select_phrase_classes_dd(session_phrase_dd, classes, ngram = 'bigram'):
    """"""
    # valid session bigrams
    valid_phrase_dd = session_phrase_dd[session_phrase_dd['_classify'].isin(classes)]
    
    if ngram == 'unigram':
        # valid session unigram conversion
        valid_phrase_dd['phrase'] =  valid_phrase_dd['phrase'].str.split(' ')

    return valid_phrase_dd


def mentions(df):
    """Takes in a dataframe with phrase 
        and occurence count columns and returns flat list 
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


def speaker_docs_dd(valid_phrase_dd):
    """"""
    mentions_dd = mentions(valid_phrase_dd)
    speaker_phrases = mentions_dd.groupby('speakerid')['phrase'].sum()
    
    return speaker_phrases