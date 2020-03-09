#===================================#
#=*= Preprocessing Speech Module =*=#
#===================================#

# Dependencies
import re
import os
import nltk
import pandas as pd

from functools import partial 
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords

from constant import DATA_PATH, HB_PATH, SPEECHES
from preprocess_constant import manual_stopwords, us_states, additional_stopwords

# NTLK tools
stemmer = SnowballStemmer("english")

# Regexs for Preprocessing
NON_ALPHA_NUM = "[^a-zA-z0-9\s]"
NON_LOWER_ALPHA = "[^a-z\s]"
DIGIT = "\d"
NUM = "\d+"

# Stopword Series
us_states_stopwords = " ".join(us_states).lower().split(" ")
english_stopwords = re.sub(NON_LOWER_ALPHA, "", " ".join(stopwords.words('english'))).split(" ")
name_stopwords = pd.read_csv(os.path.join(DATA_PATH, "voteview/congress_names.csv")).name.tolist()


# Stopword compiler
def stopword_regex(stopwords):
    """Compile a regular expression to match any of the words in stopwords"""
    stopword_pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
    
    return stopword_pattern

# default stopword matcher using all 3 stopword series
DEFAULT_STOPWORD = stopword_regex(
    english_stopwords + 
    manual_stopwords + 
    additional_stopwords +
    us_states_stopwords + 
    name_stopwords)


#=*= Functions for processing documents =*=#

def basic_preprocess(text):
    """
    Basic preprocessing for NLP
    
    - lowercasing
    - removing non-alphanumeric chars
    - normalizing numbers
    - removing leading and trailing whitespace
    """
    
    text = re.sub(NON_ALPHA_NUM, "", text.lower())
    
    if bool(re.search(DIGIT, text)):
        text = re.sub(NUM, "", text)

    text = text.strip()
    
    return text


def dense_preprocess(text, stopword=DEFAULT_STOPWORD):
    """
    Thorough preprocessinng for NLP
    
    - basic preprocessing
    - stopword removal 
    - whitespace removal
    - stemming
    """
    
    text = basic_preprocess(text)
    text = re.sub(stopword, "", text)
    text = list(map(stemmer.stem, text.strip().split(" ")))
    text = " ".join(text)

    return text


def preprocess_session(s, preprocess_func, local_path):
    """
    Preprocesses session _s_ with _preprocess_func_ and saves to _local_path_
    """
    # define file paths
    in_file_path = os.path.join(HB_PATH, SPEECHES % s)
    out_file_path = os.path.join(local_path, SPEECHES % s)
    
    # read file, and preprocess it
    df = pd.read_csv(in_file_path, sep="|")
    df["speech"] = list(map(preprocess_func, df["speech"].values))
    
    # write 
    df.to_csv(out_file_path, sep="|", index=False)
    
    print("Session", s, "PROCESSED")


def make_session_preprocessor(preprocess_func, local_path):
    """
    Returns function for preprocessing a specch and saving it using
    the preprocessor of your choice
    """
    preprocessor = partial(preprocess_session, 
                           preprocess_func=preprocess_func,
                           local_path=local_path)
   
    return preprocessor
    
    
    
    