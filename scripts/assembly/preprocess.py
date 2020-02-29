#===================================#
#=*= Preprocessing Speech Module =*=#
#===================================#

# Methods for preprocessing the speech text

# Dependencies
import re
import os
import nltk

from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords

# NTLK tools
stemmer = SnowballStemmer("english")
english_stopwords = stopwords.words('english')

# Regexs for Preprocessing
ALPHA_NUM = "[^a-zA-z0-9\s]"
DIGIT = "\d"
NUM = "\d+"

# Stopword Types
manual_stopwords = ['absent','committee','gentlelady','hereabout','hereinafter','hereto','herewith' 'nay',
'pro','sir','thereabout','therebeforn','therein','theretofore','therewithal','whereat','whereinto','whereupon',
 'yea','adjourn','con','gentleman','hereafter','hereinbefore','heretofore','month','none','republican','speak',
 'thereafter','thereby','thereinafter','thereunder','today','whereby','whereof','wherever','yes','ask','democrat',
 'gentlemen','hereat','hereinto','hereunder','mr','now','say','speaker','thereagainst','therefor','thereof',
 'thereunto','whereabouts','wherefore','whereon','wherewith','yield','can','etc','gentlewoman','hereby','hereof',
 'hereunto','mrs','part','senator','tell','thereat','therefore','thereon','thereupon','whereafter','wherefrom',
 'whereto','wherewithal','chairman','gentleladies','gentlewomen','herein','hereon','hereupon','nai','per','shall',
 'thank','therebefore','therefrom','thereto','therewith','whereas','wherein','whereunder','will']

us_states = ["Alabama","Alaska","Arizona","Arkansas","California","Colorado",
  "Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois",
  "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland",
  "Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana",
  "Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York",
  "North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
  "Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah",
  "Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"]

us_states_stopwords = " ".join(us_states_stopwords).lower().split(" ")

# stopword compiler
def stopword_regex(stopwords):
    """Compile a regular expression to match any of the words in stopwords"""
    
    return re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')



def basic_preprocess(text):
    """lowercasing, removing non-alphanumeric chars, and normalizing numbers"""
    
    text = text.lower()
    text = re.sub(ALPHA_NUM, '', text)
    
    if bool(re.search(DIGIT, text)):
        text = re.sub(NUM, "number", text)

    return text


def lda_preprocess(text):
    """basic preprocessing, and then stemming, tokenizing and stopword removal"""
    
    text = basic_preprocess(text)
    text = stemmer.stem(text)
    text = text.split()
    text = list(filter(lambda w: w not in english_stopwords, text))
    
    return text
