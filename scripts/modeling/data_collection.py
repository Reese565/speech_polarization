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

from constant import DOC_ALL_PATH, MIN_SESSION, MAX_SESSION
from document import load_documents
from subject import subject_keywords

from rmn import RMN
from rmn_analyzer import *

# constants
RMN_NAME = "full"
RMN_PATH = "/home/rocassius/gen-data/models"
SAVE_PATH = '/home/rocassius/gen-data/data/div-first'
DIV_TAG = 'div_data_%s.txt'

sessions = list(range(MIN_SESSION, MAX_SESSION+1))

def analyze_session(session, subjects, sample_n, doc_path, rmn):
    
    # read in session
    df = load_documents(sessions=[session], read_path=doc_path)

    # analyze
    analyzer = RMN_Analyzer(rmn, df)
    print("Analyzing Session %s ..." % format(session, '03d'))
    data = analyzer.analyze(subjects, sample_n)
    print("Data Gathered for Session %s. " % format(session, '03d'))

    # add session number
    data.update({SESS: session})

    # Save 
    with open(os.path.join(SAVE_PATH, DIV_TAG % format(session, '03d')), 'w') as outfile:
        json.dump(data, outfile)
    
    return data


def main():
    
    # time it
    start = time.time()
    
    # make rmn
    rmn = RMN(); rmn.load_rmn(name=RMN_NAME, save_path=RMN_PATH)
    
    # declare analyzing function
    analyze_func = partial(
        analyze_session, 
        subjects=subject_keywords.keys(), 
        sample_n=1000, 
        doc_path=DOC_ALL_PATH,
        rmn=rmn)
    
    # gather data
    data = [analyze_func(s) for s in sessions]
    
    # Save 
    with open(os.path.join(SAVE_PATH, DIV_TAG % 'all'), 'w') as outfile:
        json.dump(data, outfile)

    end = time.time()
    elapsed = end - start

    # report
    print("SUCCESS, took", round(elapsed / 60, 2), "minutes")
    
    
if __name__ == "__main__":
    main()
