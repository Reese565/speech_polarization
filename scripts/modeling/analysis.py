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


epsilon = 1e-10
def shannon_entropy(p):
    p = np.asarray(p)
    return np.sum(p*-np.log2(p+epsilon), axis=-1)


def mean_CI(x, as_dict=True):
    """
    Returns a numpy array of length 3 with indices as follows:
    - 0, the estimated mean
    - 1, the 95% lower CI bound
    - 2, the 95% upper CI bound
    """
    # handle nan values
    x = np.array(x)[~np.isnan(x)]
    mean = np.mean(x)
    CI = sms.DescrStatsW(x).tconfint_mean()
    mean_ci = np.array((mean,) + CI)
    
    if as_dict:        
        return {'mean':  mean_ci[0],
                'lower': mean_ci[1],
                'upper': mean_ci[2]}
    else:
        return mean_ci


    
def mean_pair_similarity(vectors):
    
    n_vec = vectors.shape[0]
    index_A = np.tile(range(n_vec), n_vec)
    index_B = np.repeat(range(n_vec), n_vec)
    
    v_A, v_B = vectors[index_A], vectors[index_B]
    norm = (np.linalg.norm(v_A, axis=1) * 
            np.linalg.norm(v_B, axis=1))
    sim = (v_A * v_B).sum(axis=1) / norm
    
    coherence = (sim.sum() - n_vec) / (n_vec**2 - n_vec)
    
    return coherence
    
    
def vector_coherence(index, W):    
    return mean_pair_similarity(W[index])
    

def word_coherence(words, word_index, W):
    return vector_coherence([word_index[w] for w in words], W)


def random_coherence(k, W):
    
    v = np.random.uniform(low=-1., high=1., size=W.shape[1])
    nn, sim = find_nn_cos(v, W, k)    
    nn_coherence = vector_coherence(nn, W)
    
    return nn_coherence    