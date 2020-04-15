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
    
from constant import HB_PATH, MIN_SESSION, MAX_SESSION, DOCUMENT
from subject import subject_keywords
from document import *


# constants
N_CORES = cpu_count()
SAVE_PATH = "/home/rocassius/gen-data/doc/doc-generic"
sessions = list(range(MIN_SESSION, MAX_SESSION+1))


def make_generic_documents(
    session,  
    speech_path, 
    write_path):
    
    # get documents
    df = documents_from_session(session, speech_path)
    
    # write
    session_str = format(session, '03d')
    df.to_csv(os.path.join(write_path, DOCUMENT % session_str), sep="|", index=False)
    
    # report
    print("DOCUMENTS MADE for session %s " % session)



def main():
    
    # time it
    start = time.time()
    
    document_maker = partial(make_generic_documents, 
                             speech_path=HB_PATH, 
                             write_path=SAVE_PATH)
    
    # execute in parallel
    #with ThreadPoolExecutor(max_workers = N_CORES) as executor:
    #    executor.map(document_maker, sessions)  
    
    document_maker(81)

    end = time.time()
    elapsed = end - start

    # report
    print("SUCCESS, took", round(elapsed / 60, 2), "minutes")
     
        
if __name__ == "__main__":
    main()


