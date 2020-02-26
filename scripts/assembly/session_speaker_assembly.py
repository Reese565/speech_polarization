#=================================#
#=*= Session Speaker Assemlbly =*=#
#=================================#

# Methods for:
# - handling speaker n_grams per session
# - processing for LDA


import pandas as pd
import numpy as np
import os

from multiprocessing import Pool, cpu_count
from itertools import product, chain
from gensim.corpora import Dictionary

# Import directory constants
from constant import DATA_PATH, HB_PATH, BY_SPEAKER, SPEAKER_MAP

# constants
N_CORES = cpu_count()

# import master vocaulary with phrase classifications
phrases_classes = pd.read_csv(os.path.join(DATA_PATH, "vocabulary/master_list.txt"), sep = "|")

# create gensim dictionary out of master vocabulary
global_dct = Dictionary([list(phrases_classes.phrase.values)])



# helper function to parallelize operations
def parallelize_dataframe(df, func, n_cores=N_CORES):
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
    session_phrase_df = (speaker_phrases.merge(phrases_classes, how = "inner", on ="phrase"))
    
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


def make_doc(df):
    """
    Takes a data frame belonging to a single speaker 
    and returns a dictionary containg their phrases and their speakerid  
    """
    # Assumes every document in the df has the same speaker
    doc = {'speakerid': df.speakerid.values[0], 
           'phrase': list(chain.from_iterable(mentions(df).phrase))}
    
    return doc



def group_docs(df_list):
    """
    Takes a list of speaker data frames and returns a list of
    speaker dictionary documents
    """
    docs = list(map(lambda df: make_doc(df), df_list))
    return docs



def speaker_docs(valid_phrase_df):
    """
    Takes a dataframe of phrases and returns a list of speaker documents
    one speaker document is a dictionary of the form {speakerid, phrase}
    where phrase is the list of phrases
    """
    # create list of dataframes by speaker
    speakers = valid_phrase_df.groupby('speakerid')
    speaker_df_list = [speakers.get_group(k) for k in speakers.groups.keys()]
    
    # Split speaker df list 
    num_partitions = N_CORES
    speaker_split = [speaker_df_list[i::num_partitions] for i in range(num_partitions)]
    
    # compute in parallel
    pool = Pool(N_CORES)
    docs = list(chain.from_iterable(pool.map(group_docs, speaker_split)))
    pool.close()
    pool.join()
    
    return docs


def encode_phrases(df, dct=global_dct):
    """
    Returns the phrase codes and counts of a dataframe according to the given dictionary
    """
    phrase_codes = list(map(lambda p: dct.token2id[p], df['phrase'].values))
    phrase_counts = list(zip(phrase_codes, df['count'].values))
    
    return phrase_counts


def make_bow_doc(df):
    """
    Takes a dataframe belonging to a single speaker with the fields 'speakerid' and 'phrase_code'
    and returns a dictionary containg their phrases and their speakerid  
    """
    # Assumes every document in the df has the same speaker
    bow_doc = {'speakerid': df.speakerid.values[0], 
               'phrase_code': list(df.phrase_code.values)}
    
    return bow_doc



def speaker_bow_docs(df, dct=global_dct):
    """
    Takes a datafame with at least the field 'speakerid', 'phrase' and 'count'.
    Returns a list of dictionaries (bow documents encoded by speaker)
    """
    # Compute phrase encoding - count tuples
    df['phrase_code'] = encode_phrases(df, dct)
    
    # create list of dataframes by speaker
    speakers = df.groupby('speakerid')
    speaker_df_list = [speakers.get_group(k) for k in speakers.groups.keys()]
    
    # get bow doc by speaker
    bow_docs = list(map(make_bow_doc, speaker_df_list))
    
    return bow_docs