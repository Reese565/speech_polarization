#===============================#
#=*= Assemble Name Stopwords =*=#
#===============================#

# Script for making a csv file of congress last names for 
# for members in congresses in the 43rd or later

import re
import os
import pandas as pd

from constant import DATA_PATH


def main():
    
    LAST_NAME = "\w+(?=,)"
    MIN_SESSION = 43
    LOCAL_PATH = "/home/rocassius/w266_final/scripts/assembly/congress_names.csv"
    
    # Load voteview data, extract names at 43rd or later
    voteview = pd.read_csv(os.path.join(DATA_PATH, "voteview/congress_ideology.csv"))
    names = list(set(voteview[voteview["congress"] >= MIN_SESSION]["bioname"]))
    
    # get last names as stop words
    name_stopwords = re.findall(LAST_NAME, " ".join(names).lower())
    
    # Write to csv
    name_stopwords_df = pd.DataFrame({"name": name_stopwords})
    name_stopwords_df.to_csv(LOCAL_PATH, index=False)
    

if __name__ == "__main__":
    main()