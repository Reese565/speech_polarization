#====================#
#=*= RMN Analyzer =*=#
#====================#

# Class for analyzing an RMN

import numpy as np
from analysis import *

# constants
SUB_KEY = 'subject'
SPEAKER = 'speakerid'
PARTY = 'party'

# party constants
REP = 'R'
DEM = 'D'


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
        
    @property
    def index(self):
        return self.df.index
         
        
    def predict_topics(self, use_generator=True):
        """Computes the topic predictions for all observations
        """
        if use_generator:
            self.topic_preds = self.rmn.predict_topics_generator(self.df)
        else:
            self.topic_preds = self.rmn.predict_topics(self.df)
        

    def bool_subset(self, col, value):
        """
        Returns a boolean vector for each observation in the
        dataframe indicating whether it meets the col = value condition
        """
        assert col in self.df.columns
        return self.df[col] == value
    
    
    def sample_indices(self, indices, n):
        """Returns a SRR of the indices provided
        """
        return np.random.choice(indices, n, replace=True)
    
    
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
          
    
    def intra_party_divergence(self, party, subject, n):
        """
        Computes the intraparty divergence for a given party 
        on a given subject
        
        Args:
        - subject: (str) subject to examine
        - party  : (str) party of interest
        - n      : (int) sample size
        
        Returns: a numpy array of length 3, where index...
        - 0 is the mean divergence point estimate:
        - 1 is the lower bound of a 95% CI
        - 2 is the upper bound of a 95% CI
        """
        # ensure that the topic predictions exist
        if self.topic_preds is None:
            self.predict_topics()
        
        # find party indicies on the subject
        party_index = self.index[self.bool_subset(PARTY, party) & 
                                 self.bool_subset(SUB_KEY, subject)]
        
        # Return none if there are fewer than 2 speakers
        if self.df.loc[party_index][SPEAKER].nunique() < 2:
            return None
        
        # Sample index pairs
        index_AB = []
        while len(index_AB) < n:
            a_b = self.sample_indices(party_index, n=2)
            # include samples whose speakers are different
            if self.df.loc[a_b][SPEAKER].nunique() == 2:
                index_AB.append(a_b)
        
        index_AB = np.asarray(index_AB)
        assert index_AB.shape == (n, 2)
        
        # get indices for each group
        index_A, index_B = index_AB[:,0], index_AB[:,1]
        
        return self.compute_JS(index_A, index_B)
    
    
    def inter_party_divergence(self, subject, n):
        """
        Computes the inter party JS divergence between Republicans and Democrats
        on a given subject
        
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
        is_subject = self.bool_subset(SUB_KEY, subject)
        index_R = self.index[self.bool_subset(PARTY, REP) & is_subject]
        index_D = self.index[self.bool_subset(PARTY, DEM) & is_subject]
        
        # return None if indices are insufficient
        if len(index_R)==0 or len(index_D)==0:
            return None
        
        # sample 
        samp_index_R = self.sample_indices(index_R, n)
        samp_index_D = self.sample_indices(index_D, n)
    
        return self.compute_JS(index_R, index_D)
    
    
    def intra_party_hh(self, party, subject, n=None):
        """
        Estimates the average Herfindahl–Hirschman Index of all 
        party members in a party on a given subject
        
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
        
        # find party indicies on the subject
        party_index = self.index[self.bool_subset(PARTY, party) & 
                                 self.bool_subset(SUB_KEY, subject)]
        
        if n is None:
            samp_index = party_index
        else:
            samp_index = self.sample_indices(party_index, n)
            
        return self.compute_HH(samp_index)
        
