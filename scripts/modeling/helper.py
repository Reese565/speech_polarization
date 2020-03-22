####################
# helper functions #
####################

import pickle


def pickle_object(x, file_name):
    """
    Helper function for pickling an object
    """
    outfile = open(file_name, "wb")
    pickle.dump(x, outfile)
    outfile.close()
    

def load_pickled_object(file_name):
    """
    Helper function for loading a pickled object
    """
    infile = open(file_name, "rb")
    x = pickle.load(infile)
    infile.close()
    
    return x