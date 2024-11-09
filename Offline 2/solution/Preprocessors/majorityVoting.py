import numpy as np
from scipy import stats

class MajorityVoting :
    def __init__(self, models):
        self.models = models
    
    def custom_prediction(self, x):
        """
        Parameters: 
            x is the data against which we shall predict
        """

        # make predictions from the models
        predictions =  np.array([model.custom_predict(x) for model in self.models])

        # get the majority votes
        majority_votes, count = stats.mode(predictions, axis=0)

        # convert to 1D array
        return majority_votes.ravel()



        
