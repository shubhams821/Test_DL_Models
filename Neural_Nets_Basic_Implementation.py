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
        

        # self.input shape is (features, batch_size) -> (128, 2)
        f, b = self.input.shape
        
        # 1. Compute element-wise local gradients
        local_grad = np.where(self.input > 0, 1.0, 1e-5) # Shape: (128, 2)

        # 2. Initialize a 3D tensor of shape (batch_size, features, features)
        # This creates an individual Jacobian matrix for every sample in the batch
        grad_tensor = np.zeros((b, f, f)) # Shape: (2, 128, 128)
        
        # 3. Use advanced indexing to fill the diagonal of each batch slice
        idx = np.arange(f)
        # local_grad.T changes shape to (2, 128) to map correctly to the batch slices
        grad_tensor[:, idx, idx] = local_grad.T

        # Take the sum across the batch dimension if needed
        grad = np.sum(grad_tensor, axis=0)
        print("*"*100)
        print("RELU grad_tensor, local_grad shape :", grad_tensor.shape, local_grad.shape)
        print("input gradient shape: ", gradient.shape)
        print("RELU: input", self.input.shape)
        print("RELU: grad", grad.shape)
        print("RELU: output grad shape: ", np.matmul(grad, gradient).shape)
        print("*"*100)
        return np.matmul(grad, gradient)


class Softmax(Activation):
    def forward(self, x):
        # Numerical stability trick: subtract max to prevent overflow
        shift_x = x - np.max(x, axis=0, keepdims=True)
        self.input = shift_x
        exp_x = np.exp(shift_x)
        norm = np.sum(exp_x, axis=0, keepdims=True)

        exp_x = exp_x / norm
        self.exp_x = exp_x  # Shape: (features, batch_size) -> (2, 2)
        return exp_x

    def backward(self, gradient):
        # self.exp_x shape is (features, batch_size) -> (2, 2)
        f, b = self.exp_x.shape
        
        # 1. Initialize a 3D tensor of shape (batch_size, features, features)
        grad_tensor = np.zeros((b, f, f)) # Shape: (2, 2, 2)

        # 2. Vectorized Softmax Jacobian for each item in the batch
        # For a batch item 'b', the entry (i,j) is:
        # diag(S) - S * S.T
        for batch_idx in range(b):
            s = self.exp_x[:, batch_idx].reshape(-1, 1) # Shape: (features, 1)
            # Outer product gives the S_i * S_j matrix terms
            # np.diagflat(s) sets up the S_i * (1 - S_i) terms on the diagonal
            grad_tensor[batch_idx] = np.diagflat(s) - np.matmul(s, s.T)

        # 3. Sum across the batch dimension (axis=0) to get the final Jacobian
        grad = np.sum(grad_tensor, axis=0) # Shape: (2, 2)

        print("*"*100)
        print("Softmax grad_tensor shape :", grad_tensor.shape)
        print("input gradient shape: ", gradient.shape) # Shape: (2, 2)
        print("Softmax: input", self.input.shape)       # Shape: (2, 2)
        print("Softmax: grad", grad.shape)              # Shape: (2, 2)
        print("Softmax: output grad shape: ", np.matmul(grad, gradient).shape) # Shape: (2, 2)
        print("*"*100)

        # (2, 2) matmul (2, 2) -> outputs (2, 2)
        return np.matmul(grad, gradient)


class CrossEntropyLoss:
    def forward(self, pred, y):
        self.pred = pred
        self.y = y
        # print(pred, y, pred.shape, y.shape)
        # print(np.matmul(pred.T, y)[0][0])
        return -np.log(np.matmul(pred.T, y)[0][0])

    def backward(self):
        print("*"*100)
        print("CrossEntropyLoss: input- pred, y", self.pred.shape, self.y.shape)
        print("*"*100)
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
        self.dbias = np.sum(gradient, axis = -1).reshape(self.bias.shape)
        print("*"*100)
        print("input shape: ", x.shape)
        print("input gradient shape: ", gradient.shape)
        print("weights, bias shape: ", self.weights.shape, self.bias.shape)
        print("dweights, dbias shape: ", self.dweights.shape, self.dbias.shape)
        print("grad, return grad shape: ", grad.shape, return_grad.shape)
        print("*"*100)
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
    x = np.random.randn(dim, 3)
    y = np.array([[0, 1],[0,1], [0,1]]).reshape(2,3)
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
        print(loss, pred)










