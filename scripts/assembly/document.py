#====================================#
#=*= Identifying Documents Module =*=#
#====================================#

# Methods for finding relevant document in speech data

import os
import numpy as np
import pandas as pd

from itertools import chain

from constant import HB_PATH, SPEECHES, SPEAKER_MAP
from preprocess import *


# constants
WINDOW_DEFAULT = 50


# helper function for filtering a dataframe
def filter_df(df, func):
    """Filters the dataframe by row according to the result of func"""
    mask = df.apply(lambda row: func(row), axis = 1)
    return df[mask]


def find_topic_span(d, s, window):
    """
    Find a span related to a subject within a document.
    
    Args:
    d:      list of tokens
    s:      subject word
    window: window size
    
    Returns:
    None if s does not occur in d,
    list of tokens of length window + 1 if s does occur in d
    """
    d = np.array(d)
    arg_loc = np.argwhere(d == s).flatten().tolist()
    
    # return empty list if document does not contain the subject
    if not arg_loc: return []
    
    # locate the span
    lower, upper = (max(0, arg_loc[0] - window // 2), 
                    min(arg_loc[0] + window // 2, len(d) - 1))
    span = d[lower:upper].tolist()
    
    return span


def find_topic_spans(docs, s, window):
    """
    Turn a list of documents in to a list of spans related to a subject
    
    Args:
    docs:   list of lists of tokens
    s:      subject word
    window: window size
    
    Returns:
    A list of document spans (may be empty
    """
    spans = list(
        chain.from_iterable(map(lambda d: find_topic_span(d, s, window), docs))
    )
    
    return spans


def subject_docs(session, subject, min_length, window=WINDOW_DEFAULT):
    """
    Returns a pandas dataframe 
    """
    # read relevant session dataframes
    session_str = format(session, '03d') 
    
    speeches = pd.read_csv(os.path.join(HB_PATH, SPEECHES % session_str), sep = "|")
    speaker_map = pd.read_csv(os.path.join(HB_PATH, SPEAKER_MAP % session_str), sep = "|")
   
    # Merge on speakerid and drop rows with missing speakerids
    df = (speeches
          .merge(speaker_map[["speakerid", "speech_id"]], how = "left", on ="speech_id")
          .dropna())
    
    # preprocess the documents
    df["speech"] = df.apply(lambda row: lda_preprocess(row["speech"]), axis=1)

    # filter documents according to min length
    df = filter_df(df, lambda row: len(row["speech"]) >= min_length)

    # get document spans
    df["speech"] = df.apply(
        lambda row: find_topic_span(row["speech"], subject, window), 
        axis=1)
    
    # filter document spans for nonemptiness
    df = filter_df(df, lambda row: len(row["speech"]) > 0)
    
    return df    