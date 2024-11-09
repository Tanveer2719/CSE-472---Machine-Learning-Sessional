'''
    Softmax Function: Converts the raw output of the network into probabilities for each class

    Cross-Entropy loss:  Measures how well the predicted probability distribution (from softmax)
    matches the actual labels (targets)
'''

import numpy as np

class SoftMax:

    def _init_(self, labels):
        self.labels = labels
        self.output = None

    # Softmax(zi)= e^(zi) / ∑ e^(zi) 
    # the subtaction "np.max(input_data, axis=1, keepdims=True)" done to ensure numerical stability
    # Output contains all the probabilities for each class for each sample in the batch 
    
    '''
        Loss= - (1/n) * ∑ log(y^)
        where  y^ is the is the predicted probability of the correct class for the ith sample
        n is the total no of samples

        
        [[0.7, 0.2, 0.1],    # First sample's softmax probabilities
        [0.1, 0.8, 0.1],    # Second sample's softmax probabilities
        [0.3, 0.4, 0.3]]    # Third sample's softmax probabilities

        labels = [0, 1, 2]

        self.output[range(num_samples), labels] returns 
        [0.7, 0,8, 0.3]

        For 1st sample the correct class is 0, the probability for class0 in softmax output is 0.7 

    '''
 

    def forward(self, input_data, labels):
        # Soft max
        exp_values = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.labels = labels

        # cross entropy loss
        num_samples = input_data.shape[0]
        confidences = -np.log(self.output[range(num_samples), labels])
        return np.mean(confidences)
    
    def backward(self):
        num_samples = self.output.shape[0]

        grad_output = self.output

        # softmax ouput [0.7 , 0.1 , 0.2] for sample[0]
        # because of the below line, grad_output becomes [-.3, .1, .2]
        # We need this gradient to push the correct class output to 1, so we only subtract the 
        # correct class output 
        grad_output[range(num_samples), self.labels] -= 1
        return grad_output / num_samples        # normalization
