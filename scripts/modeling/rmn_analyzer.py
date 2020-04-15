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
TOPIC = 'topic'
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
PL = 'placebo'


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
        
    
    def predict_topics(self, use_generator=False, k_primary_topics=5):
        """Computes the topic predictions for all observations
        """
        # predict topics
        self.topic_preds = self.rmn.predict_topics(self.df, use_generator)
        
        # dataframe of first k topics for each record
        primary_topics = self.primary_topics({}, k_primary_topics)
        primary_cols = [TOPIC + str(i) for i in range(1, k_primary_topics+1)]
        primary_df = pd.DataFrame(primary_topics, columns=primary_cols)
        
        # update analyzer dataframe
        self.df = self.df.join(primary_df)
        
    
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
    
    
    def compute_JS(self, index_A, index_B, omit_topics=[], base=2):
        """
        Computes the mean pair-wise JS divergence and associated CI
        between indices in index_A and indices in index_B
        """
        p_A = np.delete(self.topic_preds[index_A], omit_topics, axis=-1)
        p_B = np.delete(self.topic_preds[index_B], omit_topics, axis=-1)
        print(p_B.shape)
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
    
    
    def topic_use_placebo_js(self, conditions={}, n=20):
        """Returns a placebo JS divergence of the R and D topic use distributions
        """
        # get original parties
        cond_index = self.cond_index(conditions)
        party = self.df.loc[cond_index][PARTY]
        
        js_list = []
        for _ in range(n):
            # permute party labels
            self.df[PARTY][cond_index] = np.random.permutation(party)
            # compute associated js
            js = self.topic_use_RD_js(conditions)
            js_list.append(js)
        
        # restore party labels
        self.df[PARTY][cond_index] = party
        
        return mean_CI(js_list)
        
    
    def topic_use_hh(self, conditions={}):
        """Returns the HH-index of the RD topic use distributions
        """
        return hh_index(self.topic_use(conditions))
    
    
    def inter_party_js(self, conditions, n, omit_topics=[]):
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
    
        return self.compute_JS(samp_index_R, samp_index_D, omit_topics)
    
    
    def group_js(self, conditions, n, omit_topics=[]):
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
        
        return self.compute_JS(index_A, index_B, omit_topics)
    
    
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
        
        
    def analyze_subset(self, conditions, n=30):
        """
        Returns a dictionary of analysis metrics for the subset 
        of records defined by the conditions.
        
        Note: It is recommended conditions be on subject
        
        Args:
        - conditions: (dict) dictionary of conditions
        - n         : (int) sample size for estimation of metrics
        
        for the entire dataset and for each subject the following are computed:
        - n_records for all, R, D
        - n_nan_records for R, D
        - hh
        - hh_R
        - hh_D
        - js_placebo
        - js_RD
        
        Returns: a dictionary of metrics
        """
        # R and D added conditions
        conditions_R = {**conditions, **{PARTY: R}}
        conditions_D = {**conditions, **{PARTY: D}}
        
        # annotation tags
        _R = '_' + R
        _D = '_' + D
        _RD = _R + D
        _PL = '_'+ PL
        
        metrics = {
            # n records in data
            N_REC:    self.n_records(conditions),
            N_REC+_R: self.n_records(conditions_R),
            N_REC+_D: self.n_records(conditions_D),
            N_NAN+_R: self.n_nan_preds(conditions_R),
            N_NAN+_D: self.n_nan_preds(conditions_D),
            # HH Topic Use Metrics
            HH:     self.topic_use_hh(conditions),
            HH+_R:  self.topic_use_hh(conditions_R),
            HH+_D:  self.topic_use_hh(conditions_D),
            # JS Topic Use Metrics
            JS+_RD: self.topic_use_RD_js(conditions),
            JS+_PL: self.topic_use_placebo_js(conditions, n),
            # Top Use Metrics
            TP:    self.topic_use(conditions, as_tuples=True),
            TP+_R: self.topic_use(conditions_R, as_tuples=True),
            TP+_D: self.topic_use(conditions_D, as_tuples=True)
        }
        
        return metrics        
        
    
        
    def analyze(self, n):
        """
        Returns a dictionary of analysis metrics at the subject level
        and at the session level (assuming self.df is the data of a
        single session).
        
        Args:
        - n: (int) sample size for estimation of metrics
        
        Returns: a dictionary of metrics
        """
        # analyze entire session dataset
        dataset_metrics = self.analyze_subset(conditions={}, n=n)
        
        # analyze by subject
        topic_metrics = {}
        for t in range(self.rmn.num_topics):
            topic_metrics[t] = self.analyze_subset({TOPIC+str(1): t}, n)
        
        metrics = {'dataset': dataset_metrics, 
                   'topics' : topic_metrics}
        
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
    
    
    def topic_use(self, conditions={}, as_tuples=False):
        """
        Returns a leaderboard of topics based on the percentage of 
        total weight given to them in all of the documents
        """
        cond_index = self.cond_index(conditions)
        topic_sums = pd.Series(np.nansum(self.topic_preds[cond_index], axis=0))
        topic_use = topic_sums.sort_values(ascending=False) / topic_sums.sum()
        
        if not as_tuples:
            return topic_use
        else:
            return list(topic_use.iteritems())
    
    
    def primary_topics(self, conditions={}, k=5):
        """Returns top k most prominent topics for documents
        """
        cond_index = self.cond_index(conditions)
        primary_topics = np.flip(np.argsort(self.topic_preds[cond_index]), axis=-1)[:,:k]
        
        return primary_topics
    
    
    def find_topic_nns(self):
        """Finds the nearest neighbors of the rmn's topics
        """
        self.topic_nns = np.array(self.rmn.inspect_topics())
      
    
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
    
    
    def sample_records(self, conditions, n=10):
    
        samp_index = self.cond_index(conditions)
        if len(samp_index) > n:
            samp_index = np.random.choice(samp_index, n)
            
        self.show_records(samp_index)
    
    
    def show_records(self, index):
        
        for rec in self.df.loc[index].itertuples():
            print(30*'=')
            print('SPEAKER:', rec.firstname, rec.lastname)
            print('PARTY:  ', rec.party)
            print('\n', rec.document, '\n')
            print('PRIMARY TOPICS:', rec.t1, rec.t2, rec.t3, rec.t4, rec.t5, '\n')
        
        