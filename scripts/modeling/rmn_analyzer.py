#====================#
#=*= RMN Analyzer =*=#
#====================#

# Class for analyzing an RMN

import numpy as np
import pandas as pd
from analysis import *

# variable constants
SUB = 'subject'
SPEAK = 'speakerid'
PARTY = 'party'
SESS = 'session'
# party constants
R = 'R'
D = 'D'
# metric constants
JS = 'js'
HH = 'hh'
EN = 'entr'
N_REC = 'n_records'
N_NAN = 'n_nan_preds'
TP = 'topic_use'


class RMN_Analyzer(object):
    """Class for Analyzing an RMN with respect to a dataset
    """
    
    def __init__(self, rmn, df):
        """
        Args:
        - rmn: (RMN) the RMN to be used for analysis
        - df : (DataFrame) the dataframe to analyze
        """
        
        'Initialization'
        self.rmn = rmn
        self.df = df.reset_index(drop=True)
        self.topic_preds = None
        self.y_preds = None
        
        self.topic_nns = None
        self.topic_coherence = None
        
        
    @property
    def index(self):
        return self.df.index
        
    
    def predict_topics(self, use_generator=False):
        """Computes the topic predictions for all observations
        """
        self.topic_preds = self.rmn.predict_topics(self.df, use_generator)
        
    
    def predict_y(self, use_generator=True):
        """Computes the sentence vector predictions for all observations
        """
        self.y_preds = self.rmn.predict_y(self.df, use_generator)
        
        
    def sample_indices(self, indices, n):
        """Returns a SRR of the indices provided
        """
        return np.random.choice(indices, n, replace=True)

    
    def bool_subset(self, col, value):
        """
        Returns a boolean vector for each observation in the
        dataframe indicating whether it meets the col == value condition
        """
        assert col in self.df.columns
        return self.df[col] == value
    
    
    def bool_index(self, conditions):
        """
        Returns a boolean vector for each observation in the
        dataframe indicating whether it meets all conditions
        
        Args:
        - conditions: (dict) dictionary of conditions
        
        Returns: 
        - pandas series of booleans indicating where all 
          of the conditions hold
        """
        # initialize bool index
        bool_index = (pd.Series(True)
                      .repeat(self.index.shape[0])
                      .reset_index(drop=True))
        
        for col, val in conditions.items():
            bool_index = bool_index & self.bool_subset(col, val)
            
        return bool_index
    
    
    def cond_index(self, conditions):
        """Returns indices of records meeting the conditions
        """
        return self.index[self.bool_index(conditions)]
    
    
    def n_records(self, conditions={}):
        """Returns the number of records meetings the conditions
        """
        return len(self.cond_index(conditions))
    
    
    def n_nan_preds(self, conditions={}):
        """Returns the number of records which have nan predictions
        """
        cond_index = self.cond_index(conditions)
        return np.isnan(self.topic_preds[cond_index].sum(axis=-1)).sum().item()
    
    
    def compute_JS(self, index_A, index_B, base=2):
        """
        Computes the mean pair-wise JS divergence and associated CI
        between indices in index_A and indices in index_B
        """
        p_A = self.topic_preds[index_A]
        p_B = self.topic_preds[index_B]
        js_list = [jensenshannon(p, q, base) for p, q in zip(p_A, p_B)]
        
        return mean_CI(js_list)
        
        
    def compute_HH(self, index):
        """
        Computes the mean HH index and associated CI between
        indices in index_A and indices in index_B
        """
        p = self.topic_preds[index]
        hh_list = [hh_index(q) for q in p]
        
        return mean_CI(hh_list)
    
    
    def topic_use_RD_js(self, conditions={}):
        """Returns the JS divergence of the R and D topic use distributions
        """
        R_topic_use = self.topic_use({**conditions, **{PARTY: R}})
        D_topic_use = self.topic_use({**conditions, **{PARTY: D}})
        
        return jensenshannon(R_topic_use, D_topic_use)
    
    
    def topic_use_hh(self, conditions={}):
        """Returns the HH-index of the RD topic use distributions
        """
        return hh_index(self.topic_use(conditions))
    
    
    def inter_party_js(self, conditions, n):
        """
        Returns the estimated inter party JS divergence and a CI.
        
        Computes the inter party JS divergence between 
        Republicans and Democrats on a given subject
        
        Args:
        - subject: (str) subject to examine
        - n      : (int) sample size
        
        Returns: a numpy array of length 3, where
        - 0 is the mean divergence point estimate:
        - 1 is the lower bound of a 95% CI
        - 2 is the upper bound of a 95% CI
        """
        # ensure that the topic predictions exist
        if self.topic_preds is None:
            self.predict_topics()
        
        # find R and D indicies on the subject
        index_R = self.cond_index({**conditions, **{PARTY: R}})
        index_D = self.cond_index({**conditions, **{PARTY: D}})
        
        # return None if indices are insufficient
        if len(index_R)==0 or len(index_D)==0:
            return None
        
        # sample 
        samp_index_R = self.sample_indices(index_R, n)
        samp_index_D = self.sample_indices(index_D, n)
    
        return self.compute_JS(samp_index_R, samp_index_D)
    
    
    def group_js(self, conditions, n):
        """
        Returns the estimated mean JS divergence and a CI
        
        Estimates the average JS divergence between any two documents of
        a group defined by the conditions. A document by speaker _i_ is 
        never compared to another document by speaker _i_.
        
        
        Args:
        - conditions: (dict) dictionary of conditions
        - n         : (int) sample size
        
        Returns: a numpy array of length 3, where index...
        - 0 is the mean divergence point estimate:
        - 1 is the lower bound of a 95% CI
        - 2 is the upper bound of a 95% CI
        """
        # ensure that the topic predictions exist
        if self.topic_preds is None:
            self.predict_topics()
        
        # find indicies of party on the subject
        cond_index = self.cond_index(conditions)
        
        # Return none if there are fewer than 2 speakers
        if self.df.loc[cond_index][SPEAK].nunique() < 2:
            return None
        
        # Sample index pairs
        index_AB = []
        while len(index_AB) < n:
            a_b = self.sample_indices(cond_index, n=2)
            # include samples whose speakers are different
            if self.df.loc[a_b][SPEAK].nunique() == 2:
                index_AB.append(a_b)
        
        index_AB = np.asarray(index_AB)
        assert index_AB.shape == (n, 2)
        
        # get indices for each group
        index_A, index_B = index_AB[:,0], index_AB[:,1]
        
        return self.compute_JS(index_A, index_B)
    
    
    def group_hh(self, conditions={}, n=None):
        """
        Returns the estimated mean HH index and a CI
        
        Estimates the average Herfindahl–Hirschman Index 
        of all records meetings the conditons.
        
        Args:
        - subject: (str) subject to examine
        - party  : (str) party of interest
        - n      : (int) sample size
        
        Returns: a numpy array of length 3, where index...
        - 0 is the mean index point estimate:
        - 1 is the lower bound of a 95% CI
        - 2 is the upper bound of a 95% CI
        """
        # ensure that the topic predictions exist
        if self.topic_preds is None:
            self.predict_topics()
        
        # indicies meeting the conditions
        cond_index = self.cond_index(conditions)
        
        # return None if indices are insufficient
        if len(cond_index)==0:
            return None
        
        if n is None:
            return self.compute_HH(cond_index)
        else:
            samp_index = self.sample_indices(cond_index, n)
            return self.compute_HH(samp_index)
        
        
    def analyze_subset(self, conditions, n):
        """
        Returns a dictionary of analysis metrics for the subset 
        of records defined by the conditions.
        
        Note: It is recommended conditions be on subject
        
        Args:
        - conditions: (dict) dictionary of conditions
        - n         : (int) sample size for estimation of metrics
        
        for the entire dataset and for each subject the following are computed:
        - n_records, n_records_R
        - n_records_D
        - js
        - js_R
        - js_D
        - js_RD
        - hh
        - hh_R
        - hh_D
        
        Returns: a dictionary of metrics
        """
        # R and D added conditions
        conditions_R = {**conditions, **{PARTY: R}}
        conditions_D = {**conditions, **{PARTY: D}}
        
        # annotation tags
        _R = '_' + R
        _D = '_' + D
        _RD = _R + D
        _TP = '_' + TP
        
        metrics = {
            # n records in data
            N_REC:    self.n_records(conditions),
            N_REC+_R: self.n_records(conditions_R),
            N_REC+_D: self.n_records(conditions_D),
            N_NAN+_R: self.n_nan_preds(conditions_R),
            N_NAN+_D: self.n_nan_preds(conditions_D),
            # JS divergence data
            JS:     self.group_js(conditions, n),
            JS+_R:  self.group_js(conditions_R, n),
            JS+_D:  self.group_js(conditions_D, n),
            JS+_RD: self.inter_party_js(conditions, n),
            # HH index data
            HH:    self.group_hh(conditions, n),
            HH+_R: self.group_hh(conditions_R, n),
            HH+_D: self.group_hh(conditions_D, n),
            # Topic Use Metrics
            HH+_TP:    self.topic_use_hh(conditions),
            HH+_TP+_R: self.topic_use_hh(conditions_R),
            HH+_TP+_D: self.topic_use_hh(conditions_D),
            JS+_TP:    self.topic_use_RD_js(conditions),
        }
        
        return metrics
    
        
    def analyze(self, subjects, n):
        """
        Returns a dictionary of analysis metrics at the subject level
        and at the session level (assuming self.df is the data of a
        single session).
        
        Args:
        - subjects: (array-like) list of subjects
        - n       : (int) sample size for estimation of metrics
        
        Returns: a dictionary of metrics
        """
        # analyze entire session dataset
        dataset_metrics = self.analyze_subset(conditions={}, n=n)
        
        # analyze by subject
        subject_metrics = {}
        for s in subjects:
            subject_metrics[s] = self.analyze_subset({SUB: s}, n)
        
        metrics = {'dataset' : dataset_metrics, 
                   'subjects': subject_metrics}
        
        return metrics
    
    
    def shannon_entropy(self, conditions={}):
        """Returns the Shannon Entropy of topic predictions meeting conditions
        """
        # ensure that the topic predictions exist
        if self.topic_preds is None:
            self.predict_topics()
        
        return shannon_entropy(self.topic_preds[self.cond_index(conditions)])
    
    
    def mean_entropy(self, conditions={}):
        """Returns the mean entropy of topic predictions meeting condiditons
        """
        return np.nanmean(self.shannon_entropy(conditions))
        
    
    def first_topic_counts(self, conditions={}):
        """
        Returns a leaderboard of topics and how many times they 
        are the primary topic associated with a document.
        """
        if self.topic_preds is None:
            self.predict_topics()
           
        cond_index = self.cond_index(conditions)
        topic_counts = pd.Series(np.argmax(self.topic_preds[cond_index], axis=-1)).value_counts()
        
        return topic_counts
    
    
    def topic_use(self, conditions={}):
        """
        Returns a leaderboard of topics based on the percentage of 
        total weight given to them in all of the documents
        """
        cond_index = self.cond_index(conditions)
        topic_sums = pd.Series(np.nansum(self.topic_preds[cond_index], axis=0))
        topic_use = topic_sums.sort_values(ascending=False) / topic_sums.sum()
        
        return topic_use
    
    
    def primary_topics(self, conditions={}, k=5):
        """Returns top k most prominent topics for documents
        """
        cond_index = self.cond_index(conditions)
        primary_topics = np.flip(np.argsort(self.topic_preds[cond_index]), axis=-1)[:,:k]
        
        return primary_topics
    
    
    def find_topic_nns(self):
        """Finds the nearest neighbors of the rmn's topics
        """
        self.topic_nns = np.array(rmn.inspect_topics())
      
    
    def find_topic_coherence(self, k=5):
        """Updates the topic coherence scores of the 
        """
        W = self.rmn.infer_embedding_matrix
        word_index = self.rmn.infer_tokenizer_dict['word_index']
        coherence_scores = [word_coherence(np.array(t)[:5,0], word_index, W) 
                            for t in self.topic_nns]
        
        self.topic_coherence = pd.Series(coherence_scores).sort_values(ascending=False)
        
    
    @property
    def topic_nn_sim(self):
        return pd.Series(self.topic_nns[:,0,1]).sort_values(ascending=False)
        