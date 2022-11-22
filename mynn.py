import numpy as np

class Network:
    def __init__(self):
        self.sequential = []

    def train(self, data, label, lr, epoch):
        for _ in range(epoch):
            # forward pass
            x = data
            for module in self.sequential:
                x = module.feedforward(x)
            
            # backward pass
            loss = label
            for module in reversed(self.sequential):
                loss = module.backpropagate(loss)
            
            # gradient descent
            for module in self.sequential:
                module.gradient_descent(lr)

    def add(self, module):
        self.sequential.append(module)

    def predict(self, data):
        x = data
        for module in self.sequential:
            x = module.feedforward(x)
        return x


class MSE:
    def __init__ (self):
        self.y_predict = 0

    def feedforward(self, z):
        self.y_predict = np.copy(z)
        return z

    def backpropagate(self, truth):
        return 2*(self.y_predict - truth)

    def gradient_descent(self, lr):
        pass

class Softmax:
    def __init__(self, no_classes):
        self.n = no_classes
        self.output = 0

    def feedforward(self, z):
        k = z - np.amax(z, axis=0)
        self.output = np.exp(k)/np.sum(np.exp(k), axis=0)
        return self.output

    def backpropagate(self, y):
        return self.output - self.onehot(y)

    def gradient_descent(self, lr):
        pass

    def onehot(self, output):
        m = np.size(output)
        onehot_output = np.zeros((self.n, m))
        onehot_output[output, np.arange(m)] = 1
        return onehot_output


class Relu:
    def __init__(self, input_size):
        self.m = input_size
        self.input = 0
        self.output = 0

    def feedforward(self, x):
        self.input = np.copy(x)
        return np.maximum(0, x)

    def backpropagate(self, d_output):
        out = np.zeros(np.shape(d_output))
        out[self.input>0] = 1
        return out*d_output

    def gradient_descent(self, lr):
        pass


class MLP:
    def __init__(self, inputs:int, outputs:int):
        self.weights = np.random.rand(outputs, inputs) - 0.5
        self.bias = np.random.rand(outputs, 1) - 0.5
        self.x = 0
        self.dw = 0
        self.db = 0

    def feedforward(self, x):
        self.x = np.copy(x)
        return np.matmul(self.weights, x) + self.bias

    def backpropagate(self, dloss):
        _, m = dloss.shape
        self.dw = np.matmul(dloss, np.transpose(self.x))/m
        self.db = np.sum(dloss, axis=1)/m
        self.db = np.reshape(self.db, (self.db.size, 1))
        return np.matmul(np.transpose(self.weights), dloss)

    def gradient_descent(self, lr):
        self.weights -= lr*self.dw
        self.bias -= lr*self.db
