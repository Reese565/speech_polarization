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
AVG_CHARS_PER_TOKEN = 7
MIN_TOKENS = 200


speaker_info_cols = [ 
    "speakerid", 
    "lastname", 
    "firstname", 
    "chamber", 
    "state", 
    "gender", 
    "party"
]

#=*= Functions for finding documents in speeches =*=#

def save_session_documents(
    session, 
    subjects, 
    speech_path, 
    write_path, 
    min_tokens=MIN_TOKENS, 
    window_tokens=MIN_TOKENS):
    
    # get documents
    df = session_subject_docs(session, subjects, speech_path, min_tokens, window_tokens)
    
    # write
    session_str = format(session, '03d')
    df.to_csv(os.path.join(write_path, DOCUMENT % session_str), sep="|", index=False)
    
    # report
    print("DOCUMENTS MADE for session %s " % session)
    

def session_subject_docs(
    session, 
    subjects, 
    speech_path, 
    min_tokens = MIN_TOKENS, 
    window_tokens = MIN_TOKENS):
    
    # load session
    session_str = format(session, '03d') 
    speeches = pd.read_csv(os.path.join(speech_path, SPEECHES % session_str), sep = "|")
    speaker_map = pd.read_csv(os.path.join(HB_PATH, SPEAKER_MAP % session_str), sep = "|")
    
    # merge
    df_rows = (speaker_map[speaker_info_cols + ["speech_id"]]
      .merge(speeches, on="speech_id", how="right")
      .drop("speech_id", axis=1)
      .dropna()
      .apply(tuple, axis=1)
      .to_list())
    
    # filter for min character length length, find spans    
    df_rows = list(filter(lambda r: len(r[-1]) > min_tokens * AVG_CHARS_PER_TOKEN, df_rows))
    
    subject_df_list = []
    for subject in subjects:
        
        span_finder = make_span_finder(subject, window=window_tokens * AVG_CHARS_PER_TOKEN)
        
        rows = list(map(lambda r: r[:-1] + (span_finder(r[-1]), ) , df_rows))
        rows = list(filter(lambda r: r[-1] is not None, rows))
        
        subject_df = pd.DataFrame(rows, columns=speaker_info_cols + ["document"])
        subject_df["subject"] = subject
        
        subject_df_list.append(subject_df)
      
    # session df
    df = pd.concat(subject_df_list) 
    df['session'] = session
        
    return df



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
    search = re.search(r'(' + r'|'.join(keywords) + ')', d)
    if not search: return None
    
    # locate the first match     found: [80,95] -> (80 - window, 95 + window)
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
  
    
# Function for getting generic documents from a session
MIN_LENGTH = 200
DOC_LENGTH = 200
EDGE_SIZE = 100


def docs_from_speech(speech, min_length, doc_length, edge_size):
    
    words = speech.split()[:-edge_size]
    if len(words) < min_length:
        return []
    
    docs = [" ".join(words[i*doc_length:(i+1)*doc_length]) 
            for i in range(len(words) // doc_length)]
    
    return docs


def documents_from_session(
    session, 
    speech_path, 
    min_length=MIN_LENGTH, 
    doc_length=DOC_LENGTH, 
    edge_size=EDGE_SIZE):
    
    # load session
    session_str = format(session, '03d') 
    speeches = pd.read_csv(os.path.join(speech_path, SPEECHES % session_str), sep = "|", skiprows=[350331])
    speaker_map = pd.read_csv(os.path.join(HB_PATH, SPEAKER_MAP % session_str), sep = "|")

    # func: speech -> List(str)
    doc_list = partial(docs_from_speech, 
                       min_length=min_length,
                       doc_length=doc_length, 
                       edge_size=edge_size)
    
    # make documents
    speeches['speech'] = speeches['speech'].map(doc_list)
    speeches.rename(columns={'speech':'document'}, inplace=True)
    
    # merge with speaker metadata
    documents = (speaker_map[speaker_info_cols + ["speech_id"]]
                 .merge(speeches, on='speech_id', how='right')
                 .drop('speech_id', axis=1)
                 .explode('document')
                 .dropna())
    
    # correct type
    documents['speakerid'] = documents['speakerid'].astype(int)

    return documents
    
    
    
#=*= Functions for loading documents =*=#  
    
def load_documents(subjects, read_path):
    """Returns a dataframe of documents belonging the the given subjects
    """
    subject_df_list = [pd.read_csv(os.path.join(read_path, DOCUMENT % s), sep ="|") for s in subjects]
    documents_df = pd.concat(subject_df_list)
    
    # correct certain types 
    documents_df['speakerid'] = documents_df['speakerid'].astype(int).astype(str)
    documents_df['congress'] = documents_df['congress'].astype(str)
    
    return documents_df
    
    
def load_documents(sessions, read_path):
    """
    Returns a dataframe of documents belonging to the given sessions
    """
    # load data
    session_strs = [format(s, '03d') for s in sessions]
    df_list = [pd.read_csv(os.path.join(read_path, DOCUMENT % s), sep = "|") for s in session_strs]
    df = pd.concat(df_list)
    
    # correct certain types
    df['speakerid'] = df['speakerid'].astype(int).astype(str)
    df['session'] = df['session'].astype(str)
    
    return df
    
    
def load_generic_documents(sessions, read_path):
    """
    Returns a dataframe of documents belonging to the given sessions
    """
    # load data
    session_strs = [format(s, '03d') for s in sessions]
    df_list = []
    for s in session_strs:
        sess_df = pd.read_csv(os.path.join(read_path, DOCUMENT % s), sep = "|")
        sess_df['session'] = int(s)
        df_list.append(sess_df)
    
    df = pd.concat(df_list)
    
    # correct certain types
    df['speakerid'] = df['speakerid'].astype(int).astype(str)
    df['session'] = df['session'].astype(str)
    
    return df    
