#========================#
#=*= Making Documents =*=#
#========================#

# makes and saves document dataframes

import time
import pandas as pd

from multiprocessing import cpu_count, Process, Pool
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor 
from functools import partial
    
from constant import GEN_HB_PATH_2, MIN_SESSION, MAX_SESSION
from subject import subject_keywords
from document import *


# constants
N_CORES = cpu_count()
SAVE_PATH = "/home/rocassius/gen-data/doc/doc-all"
sessions = list(range(MIN_SESSION, MAX_SESSION+1))

WINDOW_TOKENS = 75

def main():
    
    # time it
    start = time.time()
    
    document_maker = partial(
        save_session_documents,
        subjects=subject_keywords.keys(), 
        speech_path=GEN_HB_PATH_2, 
        write_path=SAVE_PATH,  
        window_tokens=WINDOW_TOKENS)
    
    # execute in parallel
    with ProcessPoolExecutor(max_workers = N_CORES) as executor:
        executor.map(document_maker, sessions)  

    end = time.time()
    elapsed = end - start

    # report
    print("SUCCESS, took", round(elapsed / 60, 2), "minutes")
     
        
if __name__ == "__main__":
    main()
