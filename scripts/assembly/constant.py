#==========================#
#=*= Reusable Constants =*=#
#==========================#

# dependencies
import os

# google storage bucket data paths
DATA_PATH = "gs://rwc1/data/"
HB_PATH = os.path.join(DATA_PATH, "hein-bound/")

# data file type
BY_SPEAKER = "byspeaker_2gram_%s.txt"
SPEAKER_MAP = "%s_SpeakerMap.txt"
SPEECHES = "speeches_%s.txt"


