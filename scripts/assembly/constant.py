#==========================#
#=*= Reusable Constants =*=#
#==========================#

# dependencies
import os
import numpy as np

# google storage bucket data paths
DATA_PATH = "gs://rwc1/data/"
GEN_DATA_PATH = "gs://rwc1/gen-data/"

HB_PATH = os.path.join(DATA_PATH, "hein-bound/")
GEN_HB_PATH = os.path.join(GEN_DATA_PATH, "gen-hein-bound/")
DOC_PATH = os.path.join(GEN_DATA_PATH, "doc/")
EMBEDDINGS = os.path.join(DATA_PATH, "embeddings/")


# data file type
BY_SPEAKER = "byspeaker_2gram_%s.txt"
SPEAKER_MAP = "%s_SpeakerMap.txt"
SPEECHES = "speeches_%s.txt"
DOCUMENT = "documents_%s.txt"


# all session string appendages
MIN_SESSION = 43
MAX_SESSION = 111
SESSIONS = [format(s, '03d') for s in np.arange(MIN_SESSION, MAX_SESSION + 1)]

