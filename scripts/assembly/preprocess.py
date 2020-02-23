#===================================#
#=*= Preprocessing Speech Module =*=#
#===================================#

# Methods for preprocessing the speech text


import re
import os


# Regexs for Preprocessing
ALPHA_NUM = "[^a-zA-z0-9\s]"
DIGIT = "\d"
NUM = "\d+"



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
    text = list(filter(lambda w: w not in stop_words, text))
    
    return text
