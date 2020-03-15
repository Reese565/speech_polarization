#========================#
#=*= Making Documents =*=#
#========================#

# makes and saves document dataframes

import pandas as pd

from multiprocessing import cpu_count, Process, Pool
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor 
from functools import partial
    
from constant import GEN_HB_PATH, MIN_SESSION, MAX_SESSION
from subject import subject_keywords
from document import *


# constants
N_CORES = cpu_count()
SAVE_PATH = "/home/rocassius/gen-data/doc/"
sessions = list(range(MIN_SESSION, MAX_SESSION+1))


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

#     pool = Pool(N_CORES)
#     pool.map(assemble_save_documents, subject_keywords.keys())
#     pool.close()
#     pool.join()


#     processes = [Process(target=assemble_save_documents, args=(s,)) 
#                  for s in subject_keywords.keys()]

#     for p in processes: p.start()
#     for p in processes: p.join()

    # report
    print("SUCCESS")
     
        
if __name__ == "__main__":
    main()
© 2020 GitHub, Inc.