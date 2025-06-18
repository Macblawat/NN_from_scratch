import numpy as np

class HiddenLayer:
    def __init__(self, input_size, output_size):
        # our input is number of the previous layer and outputs is the number of neurons in this layer right
        #we use He initialization so that our gradients do not vanish
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, x):
        self.input = x
        self.output = np.dot(x, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate):
        # output_gradient is dL/dout from next layer
        weights_gradient = np.dot(self.input.T, output_gradient)
        biases_gradient = output_gradient.sum(axis=0, keepdims=True)

        # Update weights and biases
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient

        # Return gradient for previous layer
        return np.dot(output_gradient, self.weights.T)


class ReLU:
    def forward(self, x):
        self.input = x
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)