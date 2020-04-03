#=========================#
#=*= Examine Documents =*=#
#=========================#

# class to examine documents

import re
import numpy as np

from collections import defaultdict
from subject import subject_keywords


class DocumentExaminier(object):
    
    def __init__(self, df):
        """
        Args:
        - df : (DataFrame) a document dataframe
        """
        
        'Initialization'
        self.df = df.reset_index(drop = True)
        self.subjects = df.subject.unique().tolist()
        self.found_keywords = defaultdict(lambda: 
                                defaultdict(lambda:
                                defaultdict(lambda: 
                                defaultdict(lambda: 
                                defaultdict(lambda:
                                defaultdict(list))))))
        self.init_KeyWMap()
    
    
    def dd(self):
        return defaultdict(list)
        
    
    def subjectKeywordSearch(self, r, subject):
        """"""
        found = []
        found_any = False
        
        for k in subject_keywords[subject]:
            if bool(re.search(r'\b' + k, r.document)):
                found.append(k)
                found_any = True
                if not (k in self.found_keywords['subjects'][subject]['keywords'].keys()):
                    self.found_keywords['subjects'][subject]['keywords'][k]['results'] = [0,0,0]
                self.found_keywords['subjects'][subject]['keywords'][k]['index'][r.Index] = None
            
        if found_any:
            return found
        else:
            return found_any
        
        
    def init_KeyWMap(self):
        """"""
        self.found_keywords['results'] = [0,0,0]
        for s in self.subjects:
            self.found_keywords['subjects'][s]['results'] = [0,0,0]


    def examine_document(self, r):
        """"""
        print(100*"=", "\n")
        print("Example %d of %d\n" % (r.Index, self.df.shape[0]))
        print("SUBJECT:", r.subject)
        print("SESSION:", r.session)
        print("CHAMBER:", r.chamber)
        print("PARTY:", r.party)
        print("STATE:", r.state)
        print("GENDER:", r.gender,'\n')
        print('KEYWORDS:')
        for s in self.subjects:
            found = self.subjectKeywordSearch(r, s)
            if found:
                print('  ',s, '-', ', '.join(found))
        print("\n")
        print(r.document)
        print("\n")

    def examine_documents(self,):
        """"""
        for r in self.df.itertuples():
            self.examine_document(r)

   