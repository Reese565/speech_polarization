#====================================#
#=*= Identifying Documents Module =*=#
#====================================#

# Methods for finding relevant document in speech data

import numpy as np
# from itertools import chain



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
    span = list(d[lower:upper])
    
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
    