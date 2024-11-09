import numpy as np

class CustomLogisticRegression:
    
    # constructor
    def __init__(self, learning_rate=0.01, num_iterations=1500, lambda_=0.001, class_weight=None) :
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.lambda_ = lambda_
        self.class_weight = class_weight

     # sigmoid function with np.clip to prevent overflow
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
        
    def initialize_parameters(self, num_features):
        self.weights = np.zeros(num_features)
        self.bias = 0

    def update_parameters(self, d_weights, d_bias):
        self.weights -= self.learning_rate * d_weights
        self.bias -= self.learning_rate * d_bias 


    # Loss Calculation
    def compute_loss(self, y, prediction):
        # binary cross entropy
        # 1e-9 ---> to avoid taking the logarithm of zero or division by zero
        y_zero_loss = y * np.log(prediction + 1e-9)
        y_one_loss = (1-y) * np.log(1 - prediction + 1e-9)

        if self.class_weight:
            weight_0 = self.class_weight[0]
            weight_1 = self.class_weight[1]
            weighted_loss = -np.mean(weight_1* y_zero_loss + weight_0 * y_one_loss)
        else:
            weighted_loss = -np.mean(y_zero_loss + y_one_loss)
        
        # compute the L2-reularization(Ridge regulation)
        # division by len(y) inorder to normalize 
        l2_regularization = (self.lambda_ / (2*len(y)))*np.sum(np.square(self.weights))
        return  l2_regularization + weighted_loss  
    
     
    def compute_gradients(self, x, y, prediction):
        # derivative of binary cross entropy
        error =  prediction - y
        d_bias = np.mean(error)

        if self.class_weight:
            weight_0 = self.class_weight[0]
            weight_1 = self.class_weight[1]
            weights = np.where(y == 1, weight_1, weight_0)
            error = error * weights

        # compute the L2-regularization
        # since gradient is the derivative of the loss function, so L2 regularization becomes: 
        # lambda/m * w
        d_weights = np.dot(x.T, error) / len(y)
        d_weights = d_weights + ((self.lambda_/len(y)) * self.weights)
        
        return d_weights, d_bias
    
    # main loop where the model computes loss, updates its weights and bias using gradient
    # descent
    def custom_fit(self, x, y, num_iterations=None):

        num_features = x.shape[1]                   # get the number of columns in features
        self.initialize_parameters(num_features)    # initialize weights and bias
        
        if num_iterations:
            self.num_iterations =  num_iterations

        # training loop
        for i in range (self.num_iterations):
            
            x_dot_weights = np.dot(x, self.weights) + self.bias    # X.w
            prediction = self.sigmoid(x_dot_weights)

            # compute loss and gradient
            loss = self.compute_loss(y, prediction)
            d_weights, d_bias = self.compute_gradients(x, y, prediction)

            self.update_parameters(d_weights, d_bias)

    # prediction for test data
    def custom_predict(self, x, threshold=0.5):
        x_dot_weights = np.dot(x, self.weights) + self.bias
        predictions = self.sigmoid(x_dot_weights)

        return np.array([1 if p >threshold else 0 for p in predictions])
