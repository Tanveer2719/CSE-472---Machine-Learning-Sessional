import numpy as np

class RelU:
    def __init__(self):
        self.input = None
    
    # max(input, 0)
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)
    
    def backward(self, grad_out):
        # gradient of relu = 1 if > 0 else 0
        rel_grad = self.input > 0
        return grad_out * rel_grad 

