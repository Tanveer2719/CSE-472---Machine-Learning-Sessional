'''
    A Regularization technique    

    Throughout training, on each iteration, standard dropout consists of zeroing out some 
    fraction of the nodes in each layer before calculating the subsequent layer.

    Dropout adds noise to the network

    input data shape: (4,3)
    Dropout rate: 0.2 (meaning we keep 80% of neurons)

    random mask =   1 1 0
                    1 0 1
                    1 1 1
                    0 1 1
    
    input       =   0.5     0.2     0.3
                    0.1     0.4     0.7
                    0.6     0.8     0.2
                    0.9     0.3     0.4
    
    returned output :  
    0.5     0.2     0 
    0.1     0       0.7
    0.6     0.8     0.2
    0       0.3     0.4

'''
import numpy as np

class Dropout:
    def _init_(self, probability):
        '''
            probability: float, the probability that each element is removed.
        '''
        self.rate = probability
        self.mask = None
    
    def forward(self, input_data, Training = True):
        '''
            input_data: numpy array
            Training: bool, if False, the dropout layer behaves differently
        '''
        if Training:
            rng = np.random.default_rng(123567)
            keep_rate = 1 - self.rate
            # array of 0s and 1s, 1s with probability 1 - keep_prob as 1 0 0 or 1 1 0
            temp_array = (rng.random(input_data.shape) > (keep_rate)).astype(int)
            self.mask = (temp_array) / (self.rate)
            return input_data * self.mask
        
        else:
            return input_data

    
    def backward(self, error_vector):
        '''
            Neurons that were dropped out (set to zero) don't receive any gradient 
            because they had no contribution to the forward computation.
        '''
        return error_vector * self.mask