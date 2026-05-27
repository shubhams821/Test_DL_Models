import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

    def predict(self, x):
        return self.forward(x)










def create_batches(x, y, batch_size):
    """Create mini-batches for training"""
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, x.shape[0] - batch_size + 1, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield x[batch_indices], y[batch_indices]


def one_hot_encode(y, num_classes):
    """Convert integer labels to one-hot encoding"""
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


def calculate_accuracy(predictions, y_true):
    """Calculate classification accuracy"""
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    return np.mean(pred_labels == true_labels)


# ============================================================================
# MNIST TRAINING
# ============================================================================

def load_mnist():
    """Load and preprocess MNIST dataset"""
    print("Loading MNIST dataset...")

    # Load MNIST from sklearn
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    x, y = mnist.data.values, mnist.target.values.astype(int)

    # Normalize pixel values to [0, 1]
    x = x / 255.0

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # One-hot encode labels
    y_train_onehot = one_hot_encode(y_train, 10)
    y_test_onehot = one_hot_encode(y_test, 10)

    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Input shape: {x_train.shape[1]}")

    return x_train, y_train_onehot, x_test, y_test_onehot


def train_mnist():
    """Train neural network on MNIST"""

    # Load data
    x_train, y_train, x_test, y_test = load_mnist()

    # Build network architecture
    # Input(784) -> Dense(128) -> ReLU -> Dense(64) -> ReLU -> Dense(10) -> Softmax
    print("\nBuilding Neural Network...")
    model = NeuralNetwork()

    model.add(Dense(784, 128))
    model.add(ReLU())
    model.add(Dense(128, 64))
    model.add(ReLU())
    model.add(Dense(64, 10))
    model.add(Softmax())

    # Compile model
    model.compile(
        loss_function= CrossEntropyLoss(), optimizer= SGD()
    )

    print("Architecture:")
    print("  Input(784) -> Dense(128) -> ReLU")
    print("  Dense(128) -> Dense(64) -> ReLU")
    print("  Dense(64) -> Dense(10) -> Softmax")
    print(f"  Optimizer: SGD(lr=0.1)")

    # Training parameters
    epochs = 10
    batch_size = 128

    print(f"\nTraining for {epochs} epochs with batch size {batch_size}...")
    print("=" * 70)

    # Training loop
    for epoch in range(epochs):
        epoch_losses = []
        epoch_accuracies = []

        # Train on batches
        for x_batch, y_batch in create_batches(x_train, y_train, batch_size):
            loss, predictions = model.train(x_batch, y_batch)
            epoch_losses.append(loss)
            epoch_accuracies.append(calculate_accuracy(predictions, y_batch))

        # Calculate average metrics
        avg_loss = np.mean(epoch_losses)
        avg_accuracy = np.mean(epoch_accuracies)

        # Evaluate on test set
        test_predictions = model.predict(x_test)
        test_accuracy = calculate_accuracy(test_predictions, y_test)

        # Print progress
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {avg_accuracy:.4f}")
        print(f"  Test Acc: {test_accuracy:.4f}")
        print("-" * 70)

    print("\nTraining completed!")
    print("=" * 70)

    # Final evaluation
    final_predictions = model.predict(x_test)
    final_accuracy = calculate_accuracy(final_predictions, y_test)
    print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")

    return model


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NEURAL NETWORK FROM SCRATCH - MNIST CLASSIFICATION")
    print("=" * 70)

    # Train the model
    model = train_mnist()







