#==========================#
#=*= Evaluate Documents =*=#
#==========================#

# class to evaluate document quality

import re
import os

import numpy as np
from collections import defaultdict

from subject import subject_keywords
from examine_docs import DocumentExaminier

os.chdir("../modeling/")
from helper import pickle_object, load_pickled_object


EVAL_TAG = "eval_%s"
SaVE_PATH = '/home/reese56/w266_final/data/evals/'

class DocumentEvaluator(DocumentExaminier):
    
    def __init__(self, df):
        """
        Parent: DocumentExaminier
        
        Args:
        - df : (DataFrame) a documents dataframe
        """
        
        'Initialization'
        DocumentExaminier.__init__(self, df)
        self.evl_started = False
        self.saved_index = 0
        self.save_path = None
        self.check_pint = 0
        
    def results(self):
        """"""
        
        self.found_keywords['results'][0] = self.found_keywords['results'][1]/self.found_keywords['results'][2]
        print('Total Evaluated:', self.found_keywords['results'][2])
        print("\nTotal Comprehension Rate:", self.found_keywords['results'][0],'\n')
        
        for s in self.subjects:
            s_obs = self.found_keywords['subjects'][s]['results'][2]
            s_true = self.found_keywords['subjects'][s]['results'][1]
            if s_obs!=0:
                self.found_keywords['subjects'][s]['results'][0] = s_true/s_obs
                print("\nComprehension Rate \"%s\":" % s, np.round(s_true/s_obs,5))
            for key in self.found_keywords['subjects'][s]['keywords'].keys():
                k_obs = self.found_keywords['subjects'][s]['keywords'][key]['results'][2]
                k_true = self.found_keywords['subjects'][s]['keywords'][key]['results'][1]
                self.found_keywords['subjects'][s]['keywords'][key]['results'][0] = k_true/k_obs
                print("   Keword \"%s\":" % key, np.round(k_true/k_obs,5))
            
    
    def evaluate(self,k,s,v=None):
        """"""
        
        if v == None:
            v = input("Germaness of phrase \"%s\" in Subject \"%s\" (True/1 or False/0):" % (k, s))
            return self.evaluate(k,s,v)
        try:
            v = int(v)
        except:
            print("\nERROR: Entry must be True/1 or False/0")
            v = input("Germaness of phrase \"%s\" in Subject \"%s\" (True/1 or False/0):" % (k, s))
            return self.evaluate(k,s,v)
        if v == 0 or v == 1:
            return v
        else:
            v = 'invalid'
            return self.evaluate(k,s,v)
    
    
    
    def save(self, r, s=None):
        """"""
        if s == None:
            s = input('Save? (y/n):')
            return self.save(r,s)
        elif s == 'y':
            self.saved_index = r.Index + 1
            self.check_pint += 1
            name = input("Filename:")
            path = input("Save path:")
            self.save_eval(name, path)
            return True
        elif s == 'n':
            return False
        else:
            print('Please enter y/n:')
            s = None
            return self.save(r,s)
    
    def save_eval(self, name, save_path):
        """"""
        
        attribute_dict = {
            'subjects': self.subjects,
            'evl_started': self.evl_started,
            'saved_index': self.saved_index}

        
        # make path for eval
        eval_path = os.path.join(save_path, EVAL_TAG % name)
        os.mkdir(eval_path)
        
        # pickel df
        pickle_object(self.df, os.path.join(eval_path, 'dataframe'))
        
        # pickel found_keywords
        pickle_object(self.to_embed_dict(self.found_keywords), os.path.join(eval_path, 'found_keywords'))
        
        # pickle attributes
        pickle_object(attribute_dict, os.path.join(eval_path, 'attributes'))
        

    def load_eval(self, name, save_path):
        """
        Load the model, weights, architecture and attributes from a saved model
        """
        
        # make path for eval
        eval_path = os.path.join(save_path, EVAL_TAG % name)
        
        # load found_keywords dict
        self.found_keywords = self.to_embed_ddict(load_pickled_object((os.path.join(eval_path, 'found_keywords'))))
        
        # load found_keywords dict
        self.df = load_pickled_object((os.path.join(eval_path, 'dataframe')))

        
        # load attributes
        attributes_dict = load_pickled_object(os.path.join(eval_path, 'attributes'))
        
        attribute_dict = {
            'subjects': self.subjects,
            'evl_started': self.evl_started,
            'saved_index': self.saved_index}
        
        # update attributes
        self.subjects = attributes_dict['subjects']
        self.evl_started = attributes_dict['evl_started']
        self.saved_index = attributes_dict['saved_index']
        
        
    
    def update(self,r,k,s,v):
        """"""
        self.found_keywords['subjects'][s]['keywords'][k]['index'][r.Index] = v
        self.found_keywords['subjects'][s]['keywords'][k]['results'][1] += v
        self.found_keywords['subjects'][s]['keywords'][k]['results'][2] += 1
        self.found_keywords['subjects'][s]['results'][1] += v
        self.found_keywords['subjects'][s]['results'][2] += 1
        self.found_keywords['results'][1] += v
        self.found_keywords['results'][2] += 1
        

    def evaluate_documents(self):
        """"""
        
        if self.evl_started:
            df_rows = self.df.iloc[self.saved_index:,:]
        else:
            df_rows = self.df
            self.evl_started = True
        
        halt = False
        
        for i,r in enumerate(df_rows.itertuples()):
            self.examine_document(r)
            found_keys = self.subjectKeywordSearch(r, r.subject)
            if found_keys:
                print('\n--- LABELED SUBJECT KEYWORDS ---')
                for k in found_keys:
                    v = self.evaluate(k,r.subject)
                    self.update(r,k,r.subject,v)
            print('\n--- CROSS SUBJECT KEYWORDS ---')
            for s in self.subjects:
                if s != r.subject:
                    cross_keys = self.subjectKeywordSearch(r, s)
                    if cross_keys:
                        for k in cross_keys:
                            v = self.evaluate(k, s)
                            self.update(r,k,s,v)
            if i%5==0:
                halt = self.save(r)
                if halt:
                    print(100*"=", "\n")
                    print('\nProgress Saved and Stored')
                    print('Completed %d of %d examples' % (r.Index, self.df.shape[0]))
                    print("Last Index:", self.saved_index,'\n')
                    break


                    
                    
                    
        print(100*"=", "\n")
        if not halt:
            print('Evaluation Complete!')
        self.results()


    def to_embed_dict(self,d):
        """Converts an embeded object of dept 6 to a default dict"""
        d = dict(d)

        for k in d.keys():
            if k != 'results':
                d[k] = dict(d[k])
                for j in d[k].keys():
                    if j != 'results':
                        d[k][j] = dict(d[k][j])
                        for l in d[k][j].keys():
                            if l != 'results':
                                d[k][j][l] = dict(d[k][j][l])
                                for f in d[k][j][l].keys():
                                    if f != 'results':
                                        d[k][j][l][f] = dict(d[k][j][l][f])
                                        for g in d[k][j][l][f].keys():
                                            if g != 'results':
                                                d[k][j][l][f][g] = dict(d[k][j][l][f][g])

        return d


    def to_embed_ddict(self,d):
        """Converts an embeded object of dept 6 to a default dict"""
        d = defaultdict(lambda: defaultdict(lambda:
                                defaultdict(lambda: 
                                defaultdict(lambda: 
                                defaultdict(lambda:
                                defaultdict(list))))),d)

        for k in d.keys():
            if k != 'results':
                d[k] = defaultdict(lambda:defaultdict(lambda: 
                                          defaultdict(lambda: 
                                          defaultdict(lambda:
                                          defaultdict(list)))),d[k])
                for j in d[k].keys():
                    if j != 'results':
                        d[k][j] = defaultdict(lambda: defaultdict(lambda: 
                                                      defaultdict(lambda:
                                                      defaultdict(list))),d[k][j])
                        for l in d[k][j].keys():
                            if l != 'results':
                                d[k][j][l] = defaultdict(lambda: defaultdict(lambda:
                                                                 defaultdict(list)),d[k][j][l])
                                for f in d[k][j][l].keys():
                                    if f != 'results':
                                        d[k][j][l][f] = defaultdict(lambda:defaultdict(list),d[k][j][l][f])
                                        for g in d[k][j][l][f].keys():
                                            if g != 'results':
                                                d[k][j][l][f][g] = defaultdict(list,d[k][j][l][f][g])

        return d


