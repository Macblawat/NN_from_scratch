class NeuralNetwork:
    def __init__(self):
        #we initiate the neural network with an empty list of layers
        self.layers = []
    def add(self, layer):
        # we can add a layer ( we classify activation functions as layers also)
        self.layers.append(layer)

    def forward(self, x):
        #we get the input with (batch_size,number_of_features), for every value from input we apply the weights and biases from the layers or activation function
        # so with every iteration our x is smaller as we approach the output and our final x is our final probabilities
        for layer in self.layers:
            x = layer.forward(x)
        return x



