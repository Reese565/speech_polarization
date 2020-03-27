#================# 
#=*= Analysis =*=#
#================# 


# Module for analysis of probability distributions

# Dependencies
import numpy as np
import statsmodels.stats.api as sms

from scipy.special import rel_entr



def jensenshannon(p, q, base=None):
    """
    Returns the JS divergence between two 1-dimensional probability vectors,
    code taken from scipy and modified to fix bug
    """
    p = np.asarray(p)
    q = np.asarray(q)
    p = p / np.sum(p, axis=0)
    q = q / np.sum(q, axis=0)
    m = (p + q) / 2.0
    left = rel_entr(p, m)
    right = rel_entr(q, m)
    js = max(0, np.sum(left, axis=0) + np.sum(right, axis=0))
    if base is not None:
        js /= np.log(base)
    return np.sqrt(js / 2.0)


def hh_index(p):
    """
    Computes the Herfindahl–Hirschman Index for a probability distribution
    """
    p = np.asarray(p)
    p = p / np.sum(p)
    hhi = np.sum(p**2)
    return hhi


def mean_CI(x, as_dict=True):
    """
    Returns a numpy array of length 3 with indices as follows:
    - 0, the estimated mean
    - 1, the 95% lower CI bound
    - 2, the 95% upper CI bound
    """
    mean = np.mean(x)
    CI = sms.DescrStatsW(x).tconfint_mean()
    mean_ci = np.array((mean,) + CI)
    
    if as_dict:        
        return {'estimate': mean_ci[0],
                'lower':    mean_ci[1],
                'upper':    mean_ci[2]}
    else:
        return mean_ci

