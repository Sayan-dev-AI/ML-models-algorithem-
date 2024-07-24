import numpy as np

class NeuralNetwork:
  def __init__(self, input_size, hidden_size, output_size):
    # Initialize weights and biases with random values
    self.weights1 = np.random.randn(input_size, hidden_size)
    self.bias1 = np.zeros((1, hidden_size))
    self.weights2 = np.random.randn(hidden_size, output_size)
    self.bias2 = np.zeros((1, output_size))

  def feedforward(self, X):
    # Forward pass
    hidden_layer_activation = relu(np.dot(X, self.weights1) + self.bias1)
    output_layer_activation = np.dot(hidden_layer_activation, self.weights2) + self.bias2
    return output_layer_activation

def relu(x):
  return np.maximum(0, x)

# Create a neural network
nn = NeuralNetwork(1, 3, 1)  # 1 input, 3 neurons in hidden layer, 1 output

# Sample input
height = np.array([[1.8]])  # A person with height 1.8 meters

# Make a prediction
prediction = nn.feedforward(height)

print(prediction)
