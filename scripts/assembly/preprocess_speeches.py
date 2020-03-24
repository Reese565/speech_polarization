#=============================#
#=*= Preprocessing for RMN =*=#
#=============================#

# Script applies dense preprocessing to all congressional
# speeches which will be used by LDA Models.
# reads from rwc1 bucket and write locally


import time
from functools import partial

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

from constant import SESSIONS
from preprocess import *

# constants
N_CORES = cpu_count()
LOCAL_PATH = "/home/rocassius/gen-data/gen-hein-bound-2/"
    
SESSIONS = [format(s, '03d') for s in range(43, 50)] 
print(SESSIONS)
    
def main():
    
    # time it
    start = time.time()
    
    stop_words = stopword_regex(
        manual_stopwords +
        additional_stopwords +
        us_states_stopwords +
        name_stopwords)

    # create session preprocessor
    preprocess_func = partial(dense_preprocess, stopword = stop_words) 
    preprocess_this_session = make_session_preprocessor(preprocess_func, LOCAL_PATH, parallel=False)


    # execute in parallel
    with ProcessPoolExecutor(max_workers = N_CORES) as executor:
        executor.map(preprocess_this_session, SESSIONS)       

    end = time.time()
    elapsed = end - start

    # report
    print("SUCCESS, took", round(elapsed / 60, 2), "minutes")
 
        
if __name__ == "__main__":
    main()
