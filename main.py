import model
from layers import HiddenLayer, ReLU, Softmax
from loss import CrossEntropyLoss
import numpy as np

# Build the network
nn = model.NeuralNetwork()
nn.add(HiddenLayer(784, 128))
nn.add(ReLU())
nn.add(HiddenLayer(128, 64))
nn.add(ReLU())
nn.add(HiddenLayer(64, 10))
nn.add(Softmax())
# Create some dummy input (e.g. one Fashion MNIST flattened image)
x = np.random.rand(4, 784)
loss_fn = CrossEntropyLoss()
# Forward pass
y_batch = np.array([1, 3, 7, 0])
output = nn.forward(x)
loss = loss_fn.forward(output, y_batch)
print("Output shape:", output.shape)
print("Output:", output)
print("Loss:", loss)