#===================================#
#=*= Preprocessing Speech Module =*=#
#===================================#

# Dependencies
import re
import os
import nltk

from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords

from preprocess_constant import manual_stopwords, us_states, 

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


# stopword compiler
def stopword_regex(stopwords):
    """Compile a regular expression to match any of the words in stopwords"""
    
    return re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')

# default stopword matcher using all 3 stopword series
DEFAULT_STOPWORD = stopword_regex(
    english_stopwords + 
    manual_stopwords + 
    us_states_stopwords)


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


# Bad, wont use this
def lda_preprocess(text):
    """basic preprocessing, and then stemming, tokenizing and stopword removal"""
    
    text = basic_preprocess(text)
    text = stemmer.stem(text)
    text = text.split()
    text = list(filter(lambda w: w not in english_stopwords, text))
    
    return text
