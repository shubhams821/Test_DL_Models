import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

class Activation:
    """Base class for activation functions"""
    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError


class ReLU(Activation):
    """
    ReLU activation: f(x) = max(0, x)
    Derivative: f'(x) = 1 if x > 0, else 0
    """
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, dout):
        # Gradient flows through only where input was positive
        dx = dout.copy()
        dx[self.input <= 0] = 0
        return dx


class Sigmoid(Activation):
    """
    Sigmoid activation: f(x) = 1 / (1 + e^(-x))
    Derivative: f'(x) = f(x) * (1 - f(x))
    """
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip for numerical stability
        return self.output

    def backward(self, dout):
        # Derivative of sigmoid
        return dout * self.output * (1 - self.output)


class Softmax(Activation):
    """
    Softmax activation: converts logits to probability distribution
    f(x_i) = e^(x_i) / sum(e^(x_j))
    """
    def forward(self, x):
        # Subtract max for numerical stability (doesn't change output)
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output

    def backward(self, dout):
        # When used with CrossEntropy, gradient is computed in the loss
        # This is just a pass-through
        return dout


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class CrossEntropyLoss:
    """
    Cross-Entropy Loss for multi-class classification
    L = -sum(y_true * log(y_pred))

    Combined with Softmax, gradient simplifies to: y_pred - y_true
    """
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: predictions (N, C) - probabilities after softmax
            y_true: true labels (N, C) - one-hot encoded
        Returns:
            scalar loss
        """
        self.y_pred = y_pred
        self.y_true = y_true

        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)

        # Calculate loss: -sum(y_true * log(y_pred)) / batch_size
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / y_pred.shape[0]
        return loss

    def backward(self):
        """
        Gradient of CrossEntropy + Softmax combined:
        dL/dx = (y_pred - y_true) / batch_size
        """
        batch_size = self.y_pred.shape[0]
        return (self.y_pred - self.y_true) / batch_size


# ============================================================================
# LAYERS
# ============================================================================

class Dense:
    """
    Fully Connected (Dense) Layer

    Forward: y = xW + b
    Backward:
        - dL/dW = x^T * dout
        - dL/db = sum(dout, axis=0)
        - dL/dx = dout * W^T
    """
    def __init__(self, input_size, output_size):
        """
        Initialize weights using He initialization for better gradient flow
        """
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))

        # Gradients (computed during backward pass)
        self.dweights = None
        self.dbias = None

    def forward(self, x):
        """
        Args:
            x: input (batch_size, input_size)
        Returns:
            output (batch_size, output_size)
        """
        self.input = x  # Store for backward pass
        return np.dot(x, self.weights) + self.bias

    def backward(self, dout):
        """
        Args:
            dout: gradient from next layer (batch_size, output_size)
        Returns:
            dx: gradient w.r.t input (batch_size, input_size)
        """
        # Gradient w.r.t weights: X^T * dout
        self.dweights = np.dot(self.input.T, dout)

        # Gradient w.r.t bias: sum over batch dimension
        self.dbias = np.sum(dout, axis=0, keepdims=True)

        # Gradient w.r.t input: dout * W^T (for previous layer)
        dx = np.dot(dout, self.weights.T)
        return dx


# ============================================================================
# OPTIMIZERS
# ============================================================================

class Optimizer:
    """Base class for optimizers"""
    def update(self, layers):
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent
    Updates weights: W = W - learning_rate * dW
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layers):
        """
        Update weights and biases for all layers
        """
        for layer in layers:
            if isinstance(layer, Dense):
                layer.weights -= self.learning_rate * layer.dweights
                layer.bias -= self.learning_rate * layer.dbias


# ============================================================================
# NEURAL NETWORK
# ============================================================================

class NeuralNetwork:
    """
    Main Neural Network class that orchestrates:
    - Forward propagation
    - Backward propagation
    - Weight updates
    """
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.optimizer = None

    def add(self, layer):
        """Add a layer or activation to the network"""
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        """Set loss function and optimizer"""
        self.loss_function = loss
        self.optimizer = optimizer

    def forward(self, x):
        """
        Forward pass through all layers
        """
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, dout):
        """
        Backward pass through all layers in reverse order
        """
        gradient = dout
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def update_weights(self):
        """Update weights using optimizer"""
        self.optimizer.update(self.layers)

    def train_step(self, x_batch, y_batch):
        """
        Single training step:
        1. Forward pass
        2. Calculate loss
        3. Backward pass
        4. Update weights
        """
        # Forward pass
        predictions = self.forward(x_batch)

        # Calculate loss
        loss = self.loss_function.forward(predictions, y_batch)

        # Backward pass
        gradient = self.loss_function.backward()
        self.backward(gradient)

        # Update weights
        self.update_weights()

        return loss, predictions

    def predict(self, x):
        """Make predictions (forward pass without training)"""
        return self.forward(x)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

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
        loss=CrossEntropyLoss(),
        optimizer=SGD(learning_rate=0.1)
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
            loss, predictions = model.train_step(x_batch, y_batch)
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

    print("\nModel training complete!")
    print("This implementation demonstrates:")
    print("  ✓ Manual backpropagation")
    print("  ✓ Gradient calculations")
    print("  ✓ Weight updates")
    print("  ✓ No autograd or high-level frameworks")
