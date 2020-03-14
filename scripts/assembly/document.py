#====================================#
#=*= Identifying Documents Module =*=#
#====================================#

# * Methods for finding wrangling documents into useable formats
# * Methods for finding relevant document in speech data

import os
import numpy as np
import pandas as pd

from itertools import chain
from functools import partial

from constant import HB_PATH, SPEECHES, SPEAKER_MAP, DOCUMENT
from subject import subject_keywords
from preprocess import *


# constants
AVG_CHARS_PER_TOKEN = 5
MIN_TOKENS = 50
WINDOW = MIN_TOKENS * AVG_CHARS_PER_TOKEN



#=*= Functions for finding documents in speeches =*=#

def save_subject_documents(subject, assemble_func, write_path):
    
    # get documents
    df = assemble_func(subject)
    
    # write
    df.to_csv(os.path.join(write_path, DOCUMENT % subject), sep="|", index=False)
    
    # update
    print(subject, "DOCUMENTS MADE")


def assemble_subject_docs(
    subject,
    sessions,  
    speech_path, 
    min_tokens=MIN_TOKENS, 
    window=WINDOW):
    """
    Returns a pandas dataframe of speech subsets found by 
    subject_docs_func for every session in sessions
    """
    
    get_subject_docs = partial(subject_docs,
                               span_finder=make_span_finder(subject, window),
                               speech_path=speech_path,
                               min_tokens=min_tokens)
    
    subject_df = pd.concat([get_subject_docs(s) for s in sessions])
    subject_df["subject"] = subject
    
    return subject_df


def subject_docs(
    session, 
    span_finder, 
    speech_path, 
    min_tokens):
    """Returns a pandas dataframe of speech subsets related to the inputted subject
    """
    
    # read session dataframes
    session_str = format(session, '03d') 
    speeches = pd.read_csv(os.path.join(speech_path, SPEECHES % session_str), sep = "|")
    speaker_map = pd.read_csv(os.path.join(HB_PATH, SPEAKER_MAP % session_str), sep = "|")
    
    # merge
    df = (speaker_map[["speech_id","speakerid", "party"]]
      .merge(speeches, how="right")
      .drop("speech_id", axis=1)
      .dropna()
      .to_numpy())
    
    # filter for min character length length, find spans    
    rows = filter(lambda r: len(r[-1]) > min_tokens*AVG_CHARS_PER_TOKEN, map(tuple, df))
    rows = map(lambda r: r[:-1] + (span_finder(r[-1]),) , rows)
    rows = filter(lambda r: r[-1] is not None, rows)
    
    # make dataframe, add session
    subject_df = pd.DataFrame(rows, columns=["speakerid", "party", "speech"])
    subject_df["congress"] = session
    
    return subject_df


def make_span_finder(subject, window):
    
    span_finder = partial(find_subject_span, 
                          keywords=subject_keywords[subject], 
                          window=window)
    
    return span_finder


def find_subject_span(d, keywords, window):
    """
    Find subset of a document related to subject s
    Return none if s is not in the document
    """
    
    # search
    search = re.search("(" + "|".join(keywords) + ")", d)
    if not search: return None
    
    # locate the first match
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
    span does not end or start in the middle of a word
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
  
    
#=*= Functions for loading documents =*=#  
    
def load_documents(subjects, read_path):
    """Returns a dataframe of documents belonging the the given subjects
    """
    subject_df_list = [pd.read_csv(os.path.join(read_path, DOCUMENT % s), sep ="|") for s in subjects]
    documents_df = pd.concat(subject_df_list)
    
    return documents_df
    