import numpy as np
from typing import List, Tuple, Optional


class Activation:
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, dout):
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, gradient):
        b, f = self.input.shape

        local_grad = np.where(self.input > 0, 1.0, 1e-5)

        grad_tensor = np.zeros((b, f, f))
        idx = np.arange(f)

        grad_tensor[:, idx, idx] = local_grad

        out_grad_stack = [None for i in range(b)]

        for idx in range(b):
            out_grad_stack[idx] = np.matmul(grad_tensor[idx], gradient[idx, :])
        
        final_output_grad = np.stack(out_grad_stack, axis = 0)
        # print("*"*100)
        # print("RELU grad_tensor, local_grad shape :", grad_tensor.shape, local_grad.shape)
        # print("input gradient shape: ", gradient.shape)
        # print("RELU: input", self.input.shape)
        # print("RELU: grad")
        # print("RELU: output grad shape: ", final_output_grad.shape)
        # print("*"*100)
        return final_output_grad




class Softmax(Activation):
    def forward(self, x):
        shift_x = x - np.max(x, axis = 1, keepdims = True)
        self.input = shift_x
        exp_x = np.exp(shift_x)
        norm = np.sum(exp_x, axis = 1, keepdims = True)
    

        exp_x = exp_x / norm
        self.exp_x = exp_x
        return exp_x
    

    def backward(self, gradient):
        b, f = self.exp_x.shape

        grad_tensor = np.zeros((b, f, f))

        for batch_idx in range(b):
            s = self.exp_x[batch_idx, :].reshape(-1, 1)
            grad_tensor[batch_idx] = np.diagflat(s) - np.matmul(s, s.T)
        
        out_grad_stack = [None for i in range(b)]
        for idx in range(b):
            out_grad_stack[idx] = np.matmul(grad_tensor[idx], gradient[idx, :])
        

        final_output_grad = np.stack(out_grad_stack, axis = 0)
        # print("*"*100)
        # print("Softmax grad_tensor shape :", grad_tensor.shape)       # (batch_size, features, features)
        # print("input gradient shape: ", gradient.shape)               # (features, batch_size)
        # print("Softmax: input", self.input.shape)                    # (features, batch_size)
        # print("Softmax: output grad shape: ", final_output_grad.shape) # (features, batch_size)
        # print("*"*100)
        return final_output_grad
    

class CrossEntropyLoss:
    def forward(self, pred, y):
        self.pred = np.clip(pred, 1e-15, 1.0 - 1e-15)
        self.y = y

        loss_per_sample = -np.sum(self.y * np.log(self.pred), axis = 1)

        return np.mean(loss_per_sample)
    
    def backward(self):
        batch_size = self.y.shape[0]
        return -(self.y / self.pred) / batch_size


class Dense:
    def __init__(self, in_features, out_features, bias = False):
        self.weights = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros((1, out_features))

        self.dweights = None
        self.dbias = None

    def forward(self, x):
        self.h_prev = x
        a = np.matmul(x, self.weights) + self.bias
        return a

    def backward(self, gradient):
        grad = self.weights
        # print(grad.shape, gradient.shape)
        return_grad = np.matmul(gradient, grad.T)
        self.dweights = np.matmul(self.h_prev.T, gradient )
        self.dbias = np.sum(gradient, axis = 0 ).reshape(self.bias.shape)
        # print("*"*100)
        # print("input shape: ", x.shape)
        # print("input gradient shape: ", gradient.shape)
        # print("weights, bias shape: ", self.weights.shape, self.bias.shape)
        # print("dweights, dbias shape: ", self.dweights.shape, self.dbias.shape)
        # print("grad, return grad shape: ", grad.shape, return_grad.shape)
        # print("*"*100)
        return return_grad



class SGD:
    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate
    
    def update(self, layers):
        for layer in layers:
            if isinstance(layer, Dense):
                layer.weights -= self.learning_rate * layer.dweights
                layer.bias -= self.learning_rate * layer.dbias
            
        
class NeuralNetwork:
    def __init__(self):
        self.layers: List = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def compile(self, loss_function, optimizer):
        self.loss_function = loss_function
        self.optimizer = optimizer
    
    def forward(self, x: np.array):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, dout):
        gradient = dout
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
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
    batch = 4
    x = np.random.randn(batch, dim)
    # 1. Randomly choose the number of classes between 2 and 10
    # classes = np.random.randint(2, 11)
    classes = 5
    
    # 2. Generate random true class indices for each sample in the batch
    # Example for batch=6: array([2, 0, 5, 1, 2, 4])
    random_classes = np.random.randint(0, classes, size=batch)
    
    # 3. Create a One-Hot Encoded matrix of shape (classes, batch)
    y = np.zeros((batch, classes))
    y[np.arange(batch), random_classes] = 1.0

    nn = NeuralNetwork()
    layer1 = Dense(dim, 384)
    act = ReLU()
    nn.add(layer1)
    nn.add(act)
    layer2 = Dense(384, 128)
    act = ReLU()
    nn.add(layer2)
    nn.add(act)
    layer3 = Dense(128, classes)
    act = Softmax()
    nn.add(layer3)
    nn.add(act)

    nn.compile(loss_function= CrossEntropyLoss(), optimizer= SGD())


    for i in range(10):
        loss, pred = nn.train(x, y)
        print(loss)



# Output:
"""2.7549547211140766
0.3756417865347026
0.17290348720843413
0.11864403749191667
0.09270506140443835
0.0763337030371557
0.06505767524625138
0.056719764488103314
0.05028133139734109
0.045231349149403265"""


# pred * y ->

"""array([[0.        , 0.95348158, 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.96102103],
       [0.        , 0.        , 0.        , 0.95644635, 0.        ],
       [0.        , 0.        , 0.        , 0.9521805 , 0.        ]])"""








