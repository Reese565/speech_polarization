#=============================#
#=*= Preprocessing for LDA =*=#
#=============================#

# Script applies dense preprocessing to all congressional
# speeches which will be used by LDA Models.
# reads from rwc1 bucket and write locally


import os
import numpy as np
import pandas as pd

from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor  

from preprocess import dense_preprocess, make_session_preprocessor

# constants
N_CORES = cpu_count()
LOCAL_PATH = "/home/rocassius/data/gen-hein-bound/"
# sessions = [format(s, '03d') for s in np.arange(43, 112)] # session strings
sessions = [format(s, '03d') for s in np.arange(43, 45)] 
     
    
def main():
    
    # create session preprocessor
    preprocess_session = make_session_preprocessor(
        preprocess_func=dense_preprocess,
        local_path=LOCAL_PATH)
    
    # execute in parallel
    with ThreadPoolExecutor(max_workers = N_CORES) as executor:
        executor.map(preprocess_session, sessions)    
    
    # report
    print("SUCCESS")
     
        
if __name__ == "__main__":
    main()