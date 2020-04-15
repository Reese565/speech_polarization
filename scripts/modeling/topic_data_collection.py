#=*= Script for Collecting Data =*=#

# Gathers metrics from all congressional sessions's documents
# usin an RMN Analyzer

import os
import sys
import json
import time
import pandas as pd
from functools import partial

# update paths
sys.path.append("/home/rocassius/w266_final/scripts/assembly")
sys.path.append("/home/rocassius/w266_final/scripts/modeling")

from constant import DOC_PRAYER_PATH, MIN_SESSION, MAX_SESSION
from document import load_generic_documents
from subject import subject_keywords
from helper import pickle_object

from rmn import *
from rmn_analyzer import *

# constants
RMN_NAME = "PoliteFinish"
RMN_PATH = "/home/rocassius/gen-data/models"
SAVE_PATH = '/home/rocassius/gen-data/data/topic-data-first'
TOPIC_TAG = 'topic_data_%s.pkl'


#sessions = list(range(MIN_SESSION, MAX_SESSION+1))
sample_n = 3
# sessions = list(range(65, MAX_SESSION+1))
sessions = list(range(65, 68))


def analyze_session(session, sample_n, doc_path, rmn):
    
    # read in session
    df = load_generic_documents(sessions=[session], read_path=doc_path)

    # analyze
    analyzer = RMN_Analyzer(rmn, df)
    print("Analyzing Session %s ..." % format(session, '03d'))
    analyzer.predict_topics()
    data = analyzer.analyze(sample_n)
    print("Data Gathered for Session %s. " % format(session, '03d'))

    # add session number
    data.update({SESS: session})

    # Save 
    pickle_object(data, os.path.join(SAVE_PATH, TOPIC_TAG % format(session, '03d')))

    return data



def main():
    
    # time it
    start = time.time()
    
    # make rmn
    rmn = RigidRMN(); rmn.load_rmn(name=RMN_NAME, save_path=RMN_PATH)
    
    # declare analyzing function
    analyze_func = partial(
        analyze_session, 
        sample_n=sample_n, 
        doc_path=DOC_PRAYER_PATH,
        rmn=rmn)
    
    # gather data
    data = [analyze_func(s) for s in sessions]
    
    # Save 
    pickle_object(data, os.path.join(SAVE_PATH, TOPIC_TAG % 'all'))
    
    end = time.time()
    elapsed = end - start

    # report
    print("SUCCESS, took", round(elapsed / 60, 2), "minutes")
    
    
if __name__ == "__main__":
    main()
