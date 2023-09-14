import numpy as np

class MultinomialLogisticRegression:
    def __init__(self):
        self.W = np.random.randn(10, 784)  # How to introduce Input size
        self.b = np.random.randn(10, 1)

    def one_hot_encoding(self, y):
        pass

    def linear_combination(self, x, W, b):
        return W.dot(x.T) + b

    def logit(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, y_hat):
        return (np.exp(y_hat) / (np.sum(y_hat, axis=0)))

    def loss(self, y_true, y_hat):
        """
        - Σ{[p(y_i) * log(y_hat_i)] + [1 - p(y_i) * [(1 - p(y_i)) * log(1 - y_hat_i)]}
        """

        num_samples = y_true.shape[0]
        num_classes = y_true.shape[1]

        loss = -np.sum(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))

        loss /= num_classes * num_samples

        return loss

    def derivative(self, x, y, function):
        if function == 'loss':
            return ((1 - y) / (1 - x)) - (y / x)

        elif function == 'softmax':
            return x * (1 - x)

        elif function == 'logit':
            return np.exp(-x) / ((1 - np.exp(-x)) ** 2)

    def update_parameters(self, x, y, z, y_hat, W, b, learning_rate=0.01):

        dl_db = self.derivative(y_hat, y, 'loss') * self.derivative(z, y, 'softmax')
        W -= None
        b -= None

        return W, b
