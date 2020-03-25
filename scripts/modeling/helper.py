####################
# helper functions #
####################

import pickle

from functools import partial
from multiprocessing import cpu_count, Pool


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


def parallel_list_map(lst, func, num_partitions=100, **kwargs):
    """
    Multi-threading helper to apply a function on a list. This function
    will return the same result as calling func(lst, **kwargs) directly.
    The function must take a list as input (may have other arguments) and
    return a list as its output.
    :param lst             The input list
    :param func            The function to be applied on the list
    :param num_partitions  Number of threads
    :return:               The same output list as returned by func(lst)
    """
    # Split the list based on number of partitions
    lst_split = [lst[i::num_partitions] for i in range(num_partitions)]
    # Create a thread pool
    pool = Pool(cpu_count())
    # Run the function and concatenate the result
    lst = sum(pool.map(partial(func, **kwargs), lst_split), [])
    # Clean up
    pool.close()
    pool.join()
    return lst