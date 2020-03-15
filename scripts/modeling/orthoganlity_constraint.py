import tensorflow.keras.backend as K

from tensorflow.keras.constraints import Constraint


class Orthoganal(Constraint):
    """Constrains the weight matrix of a tensor's
    hidden units to be orthogonal during optimization.
    
    # Args ---
        
        axis: axis along which orthognality condition
        is applied. Defualt of None applies to column
        orthogonality.
        
        lamb: regularization hyperparameter"""
    
    def __init__(self, lamb = 1.0, axis = 1,):
        self.axis = axis
        self.lamb = lamb

    def __call__(self, w):
        
        penalty = self.orthoganalize(w)
        
        return penalty
        
    def orthoganalize(self, w):
        
        if self.axis == 1:
            w = K.transpose(w)
            
        RR_t = K.dot(K.transpose(w), w)
        Id_mat = K.eye(int(RR_t.shape[0]))
        
        return self.lamb*K.sqrt(K.sum(K.square(RR_t - Id_mat)))
    
    