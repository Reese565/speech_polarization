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
GEN_HB_PATH_2 = os.path.join(GEN_DATA_PATH, "gen-hein-bound-2/")

DOC_PATH = os.path.join(GEN_DATA_PATH, "doc/")
DOC_ALL_PATH = os.path.join(DOC_PATH, "doc-all/")
DOC_PROPER_PATH = os.path.join(DOC_PATH, "doc-proper/")
DOC_SAMPLE_PATH = os.path.join(DOC_PATH, "doc-sample/")

TOOLS_PATH = os.path.join(GEN_DATA_PATH, "tools/")
MODEL_PATH = os.path.join(GEN_DATA_PATH, "models/")
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