import numpy as np


"""
    x = (x - mean(x)) / sqrt(var(x) + epsilon)


    y= γ * x + β
    γ and β are learnable parameters of the model
    γ is the scaling factor and β is the shifting factor


    The running mean and variance gradually converge to a stable value that reflects the 
    distribution of the entire training data, not just one mini-batch.
    During inference, the model needs a consistent reference point for normalizing the data.
    The running mean and running variance provide a stable estimation of the 
    feature distribution based on all the batches seen during training.

    Running Mean=Momentum x Previous_Running_Mean + ( 1 - Momentum) x Batch_Mean
    Running Variance = Momentum x Previous_Running_Variance + ( 1 - Momentum) x Batch_Variance
    where Momentum is a hyperparameter that determines the contribution of the new batch to the running statistics.
"""
class BatchNormalization:
    def _init_(self, input_size, momentum=0.9):
        self.input_size = input_size
        self.epsilon = np.finfo(float).eps
        self.momentum = momentum
        
        # initally all are ones, so multiplication will not affect the input
        self.gamma = np.ones((1, input_size))
        
        # initially all are zeros, so no shifting of the input 
        self.beta = np.zeros((1, input_size))
        
        # subtracting the mean and dividing by the standard deviation
        self.running_mean = np.zeros((1,input_size))
        self.running_var = np.ones((1,input_size))
        
        self.input = None
        self.batch_mean = None
        self.batch_var = None

    def forward(self, input, Training=True):
        self.input = input

        if Training:
            self.batch_mean = np.mean(input, axis=0, keepdims=True)
            self.batch_var = np.var(input, axis=0, keepdims=True)

            self.x_hat = (input - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)

            # update running mean and var
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
        
        else: # inference
            self.x_hat = (input - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        return self.gamma * self.x_hat + self.beta
    
    def backward(self, grad_output):

        # dL / dBeta = ∑ dL/dy
        grad_beta = np.sum(grad_output, axis=0, keepdims=True)

        # dL / dGamma = ∑ dL/dy * x_hat
        grad_gamma = np.sum(grad_output * self.x_hat, axis=0, keepdims=True)

        # dL / dx_hat = dL/dy * gamma
        grad_x_hat = grad_output * self.gamma

        # dL / dVar = ∑ dL/dx_hat * (x - mean) * -1/2 * (var + epsilon)^(-3/2)
        grad_var = np.sum(grad_x_hat * (self.input - self.batch_mean) * -0.5 * np.power(self.batch_var + self.epsilon, -1.5), axis=0, keepdims=True)

        # dL / dMean = ∑ dL/dx_hat * -1/sqrt(var + epsilon) + dL/dVar * ∑ -2(x - mean) / N
        grad_mean = np.sum(grad_x_hat * -1 / np.sqrt(self.batch_var + self.epsilon), axis=0) + grad_var * np.mean(-2 * (self.input - self.batch_mean), axis=0)

        N = self.input.shape[0]
        # dL / dx = dL/dx_hat * 1/sqrt(var + epsilon) + dL/dVar * 2(x - mean) / N + dL/dMean / N
        grad_input = grad_x_hat * 1 / np.sqrt(self.batch_var + self.epsilon) + grad_var * 2 * (self.input - self.batch_mean) / N + grad_mean / N


        return grad_input, grad_gamma, grad_beta