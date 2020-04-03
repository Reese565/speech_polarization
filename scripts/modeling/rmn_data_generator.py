#=========================#
#=*= RMN DataGenerator =*=#
#=========================#

# Class for creating a custom Sequence for an RMN

import numpy as np
import pandas as pd

from tensorflow.python.keras.utils.data_utils import Sequence


class RMN_DataGenerator(Sequence):
    """Generates data for an RMN"""
    
    def __init__(self, rmn, data_df, batch_size=50, shuffle=True):
        
        'Initialization'
        self.rmn = rmn
        self.data_df = data_df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = data_df.index.tolist()
        self.on_epoch_end()
        
        # shuffle indicies upon initialization
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # batch size should not exceed observations
        assert self.batch_size <= self.data_df.shape[0]
        
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indices)
         
            
    def __data_generation(self, indices):
        'Generates data containing batch_size samples' 
        # generate data for indices
        return self.rmn.prep_X(self.data_df.loc[indices], for_training=True)
    
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indices) / self.batch_size))


    def __getitem__(self, i):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = self.indices[i * self.batch_size:(i+1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indices)

        return X, y