#=============================#
#=*= Preprocessing for RMN =*=#
#=============================#

# Script applies dense preprocessing to all congressional
# speeches which will be used by LDA Models.
# reads from rwc1 bucket and write locally


import pandas as pd

from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor  

from constant import SESSIONS
from preprocess import dense_preprocess, make_session_preprocessor


# constants
N_CORES = cpu_count()
LOCAL_PATH = "/home/rocassius/data/gen-hein-bound/"
# for testing

    
def main():
    
    # create session preprocessor
    preprocess_session = make_session_preprocessor(dense_preprocess, LOCAL_PATH)
    
    # execute in parallel
    with ThreadPoolExecutor(max_workers = N_CORES) as executor:
        executor.map(preprocess_session, SESSIONS)    
    
    # report
    print("SUCCESS")
     
        
if __name__ == "__main__":
    main()