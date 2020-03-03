#=============================#
#=*= Preprocessing for LDA =*=#
#=============================#

# Script applies dense preprocessing to all congressional
# speeches which will be used by LDA Models.
# reads from rwc1 bucket and write locally


import os
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor  

from constant import HB_PATH, SPEECHES
from preprocess import dense_preprocess

# local save path
LOCAL_PATH = "/home/rocassius/data/gen-hein-bound/"

# session strings
sessions = [format(s, '03d') for s in np.arange(43, 112)]


# constants
N_CORES = cpu_count()


def preprocess_session(s):
    
    in_file_path = os.path.join(HB_PATH, SPEECHES % s)
    out_file_path = os.path.join(LOCAL_PATH, SPEECHES % s)
    
    # read file
    df = pd.read_csv(in_file_path, sep="|")
    
    # preprocess it
    df["speech"] = list(map(dense_preprocess, df["speech"].values))
    
    # write 
    df.to_csv(out_file_path, sep="|", index=False)
    
    print("Session", s, "PROCESSED")
      
  
    
def preprocess_sessions_parallel(sessions):
    
    with ThreadPoolExecutor(max_workers = N_CORES) as executor:
        executor.map(preprocess_session, sessions)
    
    
def main():
    
    preprocess_sessions_parallel(sessions)    
       
        
if __name__ == "__main__":
    main()