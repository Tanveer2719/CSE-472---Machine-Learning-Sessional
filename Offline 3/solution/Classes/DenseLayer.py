import numpy as np

class DenseLayer:

    # DenseLayer(28*28, 128)
    def __init__(self, inpt_size, outpt_size):
        # inpt_size(d) - how many features the the input data will contain
        # outpt_size(m) - how many features the output data will contain 
        # So the number of units in this layer is m

        # use the random normal distribution 
        # He initialization of weights
        self.weights = np.random.randn(inpt_size, outpt_size) * np.sqrt(2.0/inpt_size)
        self.bias = np.zeros((1, outpt_size))   # dim (m,1)
        self.input = None
        self.output = None

    def forward(self, input_data):
        self.input = input_data

        # input data dimension (n, d)
        # weights dimension (d, m)
        # dimension of self.ouput (n, m)

        self.output = np.dot(self.weights, input_data) + self.bias
        return self.output
    
    def backward(self, gradient_output, optimizer, learning_rate):

        # the self.input dimension is (n,d) ---> n samples and d features for each sample
        # we shall make m features from this d samples. 
        # So, out weight function has dim(d, m) 
        # the 'grad_output' is a parameter from the (l+1) layer to (l) layer
        # so its dimension is (n, m)
        # to compute the weight function for the (l) layer the dimension must ne (d, m)
        # So, transpose the 'self.input' (d, n) and dot with the 'grad_output' (n, m)
        # we found the gradient for the weights, (d, m)
        
        grad_weights = np.dot(self.input.T, gradient_output)
        grad_bias = np.sum(gradient_output, axis=1, keepdims=True)

        # (n, m) * (m,d)  = (n, d)
        # for back propagation to layer (l-1)
        grad_input = np.dot(gradient_output, self.weights.T)    

        self.weights  = optimizer.update(self.weights, grad_weights)
        self.bias = optimizer.update(self.bias, grad_bias)
        return grad_input

        
