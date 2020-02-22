#================#
#=*= Preclean =*=#
#================#

# Script for removing bad pipes from hein-bound speeches

import numpy as np
import re
import os


# local data path
local_data_path = "/home/rocassius/data/hein-bound"

# speech file type
SPEECHES = "speeches_%s.txt"

# regex for bad pipe (occurs within speech field)
PIPE_NOT_SEP = "(?<!\d{9})(?<!\d{10})(?<!speech_id)\\|"

# session strings
sessions = [format(s, '03d') for s in np.arange(43, 112)]


def clean_text(file_name):
    # read file
    with open(file_name, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # remove bad pipes
    content = re.sub(PIPE_NOT_SEP, "", content)
    
    # rewrite file
    with open(file_name, "w") as f:
        f.write(content)
    
    print(file_name, "CLEANED")
        
def main():

    for s in sessions:
        file_name = os.path.join(local_data_path, SPEECHES % s)
        clean_text(file_name)
        

if __name__ == "__main__":
    main()