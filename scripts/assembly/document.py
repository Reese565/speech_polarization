#====================================#
#=*= Identifying Documents Module =*=#
#====================================#

# * Methods for finding wrangling documents into useable formats
# * Methods for finding relevant document in speech data

import os
import numpy as np
import pandas as pd

from itertools import chain

from constant import HB_PATH, SPEECHES, SPEAKER_MAP
from preprocess import *


# constants
AVG_CHARS_PER_TOKEN = 5
WINDOW_DEFAULT = 50 * AVG_CHARS_PER_TOKEN


def subject_docs(
    session, 
    subject, 
    path, 
    min_len_tokens=WINDOW_DEFAULT, 
    window=WINDOW_DEFAULT):
    """Returns a pandas dataframe of speech subsets related to the inputted subject
    """
    
    # read  session dataframes
    session_str = format(session, '03d') 
    speeches = pd.read_csv(os.path.join(path, SPEECHES % session_str), sep = "|")
    speaker_map = pd.read_csv(os.path.join(HB_PATH, SPEAKER_MAP % session_str), sep = "|")
    
    # merge
    df = (speaker_map[["speakerid", "speech_id"]]
      .merge(speeches, how="right")
      .drop("speech_id", axis=1)
      .dropna())
    
    # compute minimum speech length in chars
    min_len_chars = min_len_tokens * AVG_CHARS_PER_TOKEN

    # filter for min length, find spans
    rows = filter(lambda r: len(r[1]) > min_len_chars, df.to_numpy())
    rows = map(lambda r: (r[0], find_topic_span(r[1], subject, window)), rows)
    rows = filter(lambda r: r[1] is not None, rows)
    
    subject_df = pd.DataFrame(rows, columns=["speakerid", "speech"])

    return subject_df


def find_topic_span(d, s, window):
    """
    Find subset of a document related to subject s
    Return none if s is not in the document
    """
    
    search = re.search(r"\b(" + s + r")\b\s*", d)
    
    if not search: return None
    
    # locate the match
    loc = search.span()
    
    # find lower and upper indices of span
    lower = max(0, loc[0] - window // 2) 
    upper = min(loc[1] + window // 2, len(d) - 1)
    lower, upper = adjust_span_bounds(d, lower, upper)
    
    # identify span
    span = d[lower:upper]
    
    return span


def adjust_span_bounds(d, lower, upper):
    """
    Returns the adjusted lower and upper indices of a span so that the
    span does not end in the middle of a word
    """
    
    lower -= steps_to_space(d[:lower][::-1])
    upper += steps_to_space(d[upper:])
    
    return lower, upper


def steps_to_space(d):
    """Returns the distance to nearest space or end of string"""
    
    search = re.search(" ", d)
    if not search: 
        return len(d)
    steps = search.span()[0]
    
    return steps
    