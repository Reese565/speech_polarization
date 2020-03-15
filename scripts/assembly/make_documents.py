#========================#
#=*= Making Documents =*=#
#========================#

# makes and saves document dataframes

import pandas as pd

from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor  
from functools import partial
    
from constant import GEN_HB_PATH, DOCUMENT
from subject import subject_keywords
from document import *


# constants
N_CORES = cpu_count()
SAVE_PATH = "/home/reese56/w266_final/data/gen-docs/"

# for testing
sessions = [43, 44, 45] 
GEN_HB_PATH = 'gs://rwc1/data/hein-bound/'

def main():
    
    # create assemble func which gets the dataframe of documents
    assemble_func = partial(assemble_subject_docs, sessions=sessions, speech_path=GEN_HB_PATH)
    
    # create function which applies assemble func and saves the result
    assemble_save_documents = partial(save_subject_documents, 
                                      assemble_func=assemble_func, 
                                      write_path=SAVE_PATH)
    
    # execute in parallel
    with ThreadPoolExecutor(max_workers = N_CORES) as executor:
        executor.map(assemble_save_documents, subject_keywords.keys())    
    
    # report
    print("SUCCESS")
     
        
if __name__ == "__main__":
    main()