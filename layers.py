import numpy as np

class HiddenLayer:
    def __init__(self, input_size, output_size):
        # our input is number of the previous layer and outputs is the number of neurons in this layer right
        #we use He initialization so that our gradients do not vanish
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, x):
        # we multiply the x matrix and weights matrix and add bias vector
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

class Softmax:
    def forward(self, x):
        # we subtract max for stability so that our exponent doesn't go to infinity and since we subtract it from every value it doesn't interupt the results
        exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        return self.output