import numpy as np
import pickle
import DenseLayer as dl
import Dropout as dropout
import SoftMax as softmax
import BatchNormalization as batchnorm
import importlib

importlib.reload(softmax)
importlib.reload(dropout)
importlib.reload(batchnorm)
importlib.reload(dl)

"""
    The network class is the main class that will be used to create the neural network.
    The network class will have the following methods:
        - forward: to perform the forward pass
        - backward: to perform the backward pass
        - train: to train the network
        - test: to test the network
"""

class Network:
    
    def _init_(self, layers, optimizer):
        self.layers = layers    
        self.optimizer = optimizer

    def forward(self, x, Training=True): 
        
        for layer in self.layers:
            
            # If the layer is a dropout layer, we need to pass the Training flag
            # since the dropout behaves differently in training and testing
            if isinstance(layer, dropout.Dropout):
                x = layer.forward(x, Training)
            else:
                x = layer.forward(x)
        return x
    
    
    def backward(self, grad):
        # dense layer : def backward(self, gradient_output, optimizer, learning_rate)
        # dropout layer : def backward(self, grad_output)

        for layer in reversed(self.layers):
            if isinstance(layer, dropout.Dropout):
                grad = layer.backward(grad)
            elif isinstance(layer, batchnorm.BatchNormalization):
                loss_grad, grad_gamma, grad_beta = layer.backward(grad)
                # update gamma and beta
                layer.gamma = self.optimizer.update(layer.gamma, grad_gamma)
                layer.beta = self.optimizer.update(layer.beta, grad_beta)
            else:
                grad = layer.backward(gradient_output=grad, optimizer=self.optimizer, learning_rate=.0001)

    
    def train(self, x_train, y_train, batch, itrs):
        samples = x_train.shape[0]

        for itr in range(itrs):
            
            # permutation of indices for dynamics
            indices = np.random.permutation(samples)
            x_train = x_train[indices]
            y_train = y_train[indices]

            for i in range(0, samples, batch):
                x_batch = x_train[i:i+batch]
                y_batch = y_train[i:i+batch]

                # forward pass
                y_pred = self.forward(x_batch, Training=True)
                
                # loss
                loss = softmax.SoftMax().forward(y_pred, y_batch)
                loss_grad = softmax.SoftMax().backward()

                
                # backward pass
                self.backward(loss_grad)

        print(f"iteration:  {itr+1}/{itrs}, Loss: {loss}")
                
    
    def test(self, x_test, y_test):
        y_pred = self.forward(x_test, Training=False)
        pred_labels = np.argmax(y_pred, axis=1)
        
        accuracy = np.mean(pred_labels == y_test)
        return accuracy
    
    def save_weights(self, file_name):
        weights_and_bias_data = {}
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, dl.DenseLayer):
                weights_and_bias_data[f'layer_{i}_weights'] = layer.weights
                weights_and_bias_data[f'layer_{i}_bias'] = layer.bias
            elif isinstance(layer, batchnorm.BatchNormalization):
                weights_and_bias_data[f'layer_{i}_gamma'] = layer.gamma
                weights_and_bias_data[f'layer_{i}_beta'] = layer.beta
                weights_and_bias_data[f'layer_{i}_running_mean'] = layer.running_mean
                weights_and_bias_data[f'layer_{i}_running_var'] = layer.running_var
        
        with open(file_name, 'wb') as file:
            pickle.dump(weights_and_bias_data, file)

    def reload_weights(self, file_name):
        with open(file_name, 'rb') as file:
            weights_and_bias_data = pickle.load(file)

        for i, layer in enumerate(self.layers):
            if isinstance(layer, dl.DenseLayer):
                layer.weights = weights_and_bias_data[f'layer_{i}_weights']
                layer.bias = weights_and_bias_data[f'layer_{i}_bias']
            elif isinstance(layer, batchnorm.BatchNormalization):
                layer.gamma = weights_and_bias_data[f'layer_{i}_gamma']
                layer.beta = weights_and_bias_data[f'layer_{i}_beta']
                layer.running_mean = weights_and_bias_data[f'layer_{i}_running_mean'] 
                layer.running_var = weights_and_bias_data[f'layer_{i}_running_var']


    
    