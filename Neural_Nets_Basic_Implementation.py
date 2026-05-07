



import numpy as np
from typing import List, Tuple

class Activation:
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, dout):
        raise NotImplementedError

class ReLU(Activation):
    def forward(self, x):
        self.input  = x
        return np.maximum(0, x)

    def backward(self, gradient):
        # we take gradient at dL/dh(k)
        # we calculate DH(k) / DA(k) 
        shape = self.input.shape
        grad = np.zeros(shape, shape)
        
        for i in range(len(self.input)):
            grad[i][i] = self.input[i] if self.input[i] > 0 else 0

        return np.matmul(grad, gradient)


class Softmax(Activation):
    def forward(self, x):
        self.input = x
        exp_x = np.exp(x)
        norm = np.sum(exp_x, axis = 0)

        exp_x = exp_x/norm
        self.exp_x = exp_x
        return exp_x
    
    def backward(self, gradient):
        # we take gradient at dL/dh(k)
        # we calculate DH(k) / DA(k) 

        shape = self.input.shape
        grad = np.zeros(shape, shape)

        for i in range(len(self.input)):
            for j in range(len(self.input)):
                if i != j:
                    grad[i][j] = - self.exp_x[i] * self.exp_x[j]
                else:
                    grad[i][i] = self.exp_x[i] * (1 - self.exp_x[i])
        return grad


class CrossEntropyLoss:
    def forward(self, pred, y):
        self.pred = pred
        self.y = y

        return -np.log(np.dot(pred, y))

    def backward(self):
        fx_y = np.dot(self.pred, self.y)
        return -fx_y * self.y



class Dense:
    def __init__(self, in_features, out_features, bias: bool = False):
        self.weights = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros((out_features, 1))

        self.dweights = None
        self.bias = None
    
    def forward(self, x):
        # x -> (d, 1) for now, 
        self.h_prev = x
        a = np.matmul(self.weights, x) + self.bias
        return a
    
    def backward(self, gradient):

        grad = self.weights.T

        return_grad = np.matmul(grad, gradient)

        self.dweights = (np.matmul(self.h_prev,  return_grad.T)).T
        self.dbias = return_grad
        return return_grad

class SGD:
    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate

    def update(self, layers):
        for layer in layers:
            layer = layer[0]
            if isinstance(layer, Dense):
                layer.weights -= self.learning_rate * layer.dweights
                layer.bias -= self.learning_rate * layer.dbias




class NeuralNetwork:
    def __init__(self):
        self.layers: List[Tuple[None, None]] = []


    def add(self, neuron, act):
        self.layers.append([nn, act])
    

    def compile(self, loss_function, optimizer):
        self.loss_function = loss_function
        self.optimizer = optimizer


    def forward(self, x: np.array):
        # x -> (d, 1)
        output = x
        for layer in self.layers:
            output = layer[1].forward(layer[0].forward(x))
        return output
    
    def backward(self, dout):


        gradient = dout
        for layer in reversed(self.layers):
            grad1 = layer[1].backward(gradient)
            grad2 = layer[0].backward(grad1)
            gradient = grad2
        
        return gradient

    def update(self):
        self.optimizer.update(self.layers)


    def train(self, x, y):
        pred = self.forward(x)
        loss = self.loss_function.forward(pred, y)
        gradient = self.loss_function.backward()

        self.backward(gradient)
        self.update_weights
        return loss, pred
    
    def pred(self, x):
        return self.forward(x)

















