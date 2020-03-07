#==========================#
#=*= Reusable Constants =*=#
#==========================#

# dependencies
import os
import numpy as np

# google storage bucket data paths
DATA_PATH = "gs://rwc1/data/"
HB_PATH = os.path.join(DATA_PATH, "hein-bound/")
EMBEDDINGS = os.path.join(DATA_PATH, "embeddings/")

# data file type
BY_SPEAKER = "byspeaker_2gram_%s.txt"
SPEAKER_MAP = "%s_SpeakerMap.txt"
SPEECHES = "speeches_%s.txt"

# all session string appendages
MIN_SESSION = 43
MAX_SESSION = 111
SESSIONS = [format(s, '03d') for s in np.arange(MIN_SESSION, MAX_SESSION + 1)]

