



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
        grad = np.zeros((shape[0], shape[0]))
        
        for i in range(len(self.input)):
            # grad[i][i] = self.input[i] if self.input[i] > 0 else 0
            grad[i][i] = 1 if self.input[i] > 0 else 0

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
        # print(shape)
        grad = np.zeros((shape[0], shape[0]))

        for i in range(len(self.input)):
            for j in range(len(self.input)):
                if i != j:
                    grad[i][j] = - self.exp_x[i,0] * self.exp_x[j,0]
                else:
                    grad[i][i] = self.exp_x[i,0] * (1 - self.exp_x[i,0])
        return np.matmul(grad, gradient)


class CrossEntropyLoss:
    def forward(self, pred, y):
        self.pred = pred
        self.y = y
        # print(pred, y, pred.shape, y.shape)
        # print(np.matmul(pred.T, y)[0][0])
        return -np.log(np.matmul(pred.T, y)[0][0])

    def backward(self):
        # print()
        fx_y = np.matmul(self.pred.T, self.y)[0][0]
        return -self.y / fx_y


class Dense:
    def __init__(self, in_features, out_features, bias: bool = False):
        self.weights = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros((out_features, 1))

        self.dweights = None
        self.dbias = None
    
    def forward(self, x):
        # x -> (d, 1) for now, 
        self.h_prev = x

        # print(x.shape, self.weights.shape, self.bias.shape)
        a = np.matmul(self.weights, x) + self.bias
        return a
    
    def backward(self, gradient):

        grad = self.weights.T

        return_grad = np.matmul(grad, gradient)

        self.dweights = (np.matmul(self.h_prev,  gradient.T)).T
        self.dbias = gradient
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
        self.layers.append([neuron, act])
    

    def compile(self, loss_function, optimizer):
        self.loss_function = loss_function
        self.optimizer = optimizer


    def forward(self, x: np.array):
        # x -> (d, 1)
        output = x
        for layer in self.layers:
            output = layer[1].forward(layer[0].forward(output))
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
        self.update()
        return loss, pred
    
    def pred(self, x):
        return self.forward(x)





if __name__ == "__main__":
    
    dim = 768
    x = np.random.randn(dim, 1)
    y = np.array([0, 1]).reshape(2,1)
    nn = NeuralNetwork()
    layer1 = Dense(dim, 384)
    act = ReLU()
    nn.add(layer1, act)
    layer2 = Dense(384, 128)
    act = ReLU()
    nn.add(layer2, act)
    layer3 = Dense(128, 2)
    act = Softmax()
    nn.add(layer3, act)

    nn.compile(loss_function= CrossEntropyLoss(), optimizer= SGD())
    

    for i in range(10):
        loss, pred = nn.train(x, y)
        print(loss, pred.reshape(1,2))











