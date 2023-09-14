import numpy as np
import sys
# np.set_printoptions(threshold=sys.maxsize)

class NeuralNetFromScratch:

    def __init__(self):
        """
        The Neural Net architecture
        X * W1 + b1 = Z1, g(Z1) = A1 |    A1 * W2 + b2 = Z2, g(Z2) = A2 |   A2 * W3 +b3 = Z3, h(Z3) = A3 = y_hat

        putting X through the functions is called forward propagation

        Minimizing the loss function with respects to the parameters of the Neural Network <-> Make NN estimate function
        dL(y, y_hat)/dW = derivative of loss function w.r.t. Weight matrices
        dL(y, y_hat)/db = derivative of loss function w.r.t. Bias matrices

        Minimizing the loss w.r.t. the Weight and Bias Matrices, s.t. the NN 'learns' the relationship between X and Y
        is called Backward propagation


        Main idea is that The Neural Network has weights that will be optimized
        Initial weights will be random, this means a random function will be estimated in the first iteration

        Each hidden layer will be equal to a Ai matrix which is equal to
        g(X*W1 + b1)    X:  Variable matrix (Number of observations x Number of variables) (42.000 x 784)
        g(Zi*Wi + bi)   Wi: Weight matrix (Number of variables x Number of nodes) (784 x A) A arbitrarily chosen
                        bi: Bias vector (allows neural net to shift the estimated function)
                        Ai: Non-linearized linear combination matrix of X (42.000 x A)

        Last layer will be equal to a Zi matrix which is equal to
        h(A_k * W_k+1 + b_k+1)  A_k:   Last Hidden Layer
                                W_k+1: Last Weight Matrix
                                b_k+1: Last Bias Matrix
        """

        # Neural Network Parameters to be optimized (n, m) n: Number of Rows. m: Number of columns
        self.W1 = np.random.randn(1, 784)
        self.W2 = np.random.randn(10, 1)

    def relu(self, z):
        """
        Non-Linear functions to pass the linear combinations through, such that the neural net layers will be able to
        estimate complex non-linear relationships

        MNIST data is images, where the pixel values ∈ [0, 255], so
        """

        return np.maximum(0, z)

    def softmax(self, z):
        """
        For each row (representing likelihood that the class is the i'th column for each observation (row number))
        The probability is calculated by dividing the likelihood of each element within one row
            by the sum of likelihoods of each element in the row
        So each row should represent the different options of labels for an observation
        And each column is each observation in the dataset
        """

        exps = np.exp(z)
        test = np.sum(exps, axis=0, keepdims=True)
        if np.where(test == 0)[0] == 0:
            print("This is the problem")
        softmax_values = exps / np.sum(exps, axis=0, keepdims=True)
        return softmax_values

    def forward_propagation(self, x_T, W1, W2):
        """
        Forward Propagation
        Linear combination of X made non-linear and then repeated.
        Z1 = W1 X_T
        A2 = ReLu(Z1)
        Z2 = W2 A1
        A2 = softmax(A1)
        """

        z1 = W1.dot(x_T)       # Z1 = W1 X1_T 10 x 784 * 784 x 42.000
        a1 = self.relu(z1)     # g(Z1) elements
        z2 = W2.dot(a1)        # W2 Z1 10 x 10 * 10 * 42.000
        a2 = self.softmax(z2)  # 10 x 42.000

        return z1, a1, z2, a2

    def one_hot_encoding(self, y):
        """
        0 vector with a 1 for the index of the label value
        so if y = 2  |  y = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  |  indeces:(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        one_hot_Y 42.000 x 10
        """

        one_hot_y = np.zeros((y.size, y.max() + 1))    # Make 0 vector Dim: N x 10
        one_hot_y[np.arange(y.size), y.flatten()] = 1  # Column number which is equal to class/label equals 1

        return one_hot_y

    def derivative(self, z, function_name, y=None):
        """
        Calculate the derivatives for the functions used for this Neural Networks, this includes:
        The Loss function to train the model to estimate the relationship between x and y
        The ReLu activation function for the neurons, to introduce non-linearity into the model
        The Softmax function to give a Probability Density Function of the expected output
        The Sigmoid activation function for the neurons, to see how a different activation function affects the NN
        """

        if function_name == 'relu':
            return z > 0

        elif function_name == 'softmax':
            exps = np.exp(z - z.max())
            exps / np.sum(exps, axis=0) * (1 - exps/ np.sum(exps, axis=0))
            return exps

    def back_propagation(self, x_T, y, z1, a1, z2, a2, W1, W2):
        """
        Update the weight parameters to minimize the loss function
        W = W - stepsize(learning rate) * gradient of Loss function with respects to W
        b = b - stepize(learning rate) * gradient of Loss function with respects to b
        """

        # One hot encoding y
        y = self.one_hot_encoding(y)
        y_T = y.T

        # ---- Backpropagation ----
        num_samples = y_T.size

        # Gradients of the weights and biases in the third layer
        dL_dz2 = 2 * (a2 - y_T) * self.derivative(z2, 'softmax')
        dL_dW2 = (1 / num_samples) * dL_dz2.dot(a1.T)  # 1/n s.t. gradient is averaged over all gradients in grad matrix

        # Gradients of weights and biases in the second Layer
        dL_dz1 = W2.T.dot(dL_dz2) * self.derivative(z1, 'relu')  # * is element wise multiplication not matrix multiplication
        dL_dW1 = (1 / num_samples) * dL_dz1.dot(x_T.T)

        return dL_dW1, dL_dW2,

    def update_parameters(
            self, W1, W2, dL_dW1, dL_dW2, learning_rate=0.01
    ):
        """
        Update weights and biases using Gradient Descent
        """

        W2 -= learning_rate * dL_dW2
        W1 -= learning_rate * dL_dW1

        return W1, W2

    def calculate_accuracy(self, a3, y):
        """
        This function calculates the Accuracy for the entire forecasts
        """
        y = self.one_hot_encoding(y)
        y_T = y.T
        predicted_labels = np.argmax(a3, axis=0)
        true_labels = np.argmax(y_T, axis=0)
        accuracy = np.mean(predicted_labels == true_labels)

        return accuracy

    def calculate_accuracy_per_label(self, a3, y):
        """
        This function calculates the Accuracy per label
        """
        y = self.one_hot_encoding(y)
        y_T = y.T
        predicted_labels = np.argmax(a3, axis=0)
        true_labels = np.argmax(y_T, axis=0)
        unique_labels = np.unique(true_labels)
        accuracy_per_label = {}
        for label in unique_labels:
            mask = true_labels == label
            accuracy = np.mean(predicted_labels[mask] == true_labels[mask])
            accuracy_per_label[label] = accuracy

        return accuracy_per_label

    def train_model(self, x, y, epochs=1, learning_rate=0.01):
        """
        Epoch is the number of times the neural network will get trained on the training data
        """

        # First get initial random parameters
        W1 = self.W1
        W2 = self.W2

        for epoch in range(epochs):
            # Forward Propagation
            z1, a1, z2, a2 = self.forward_propagation(
                x, W1, W2
            )

            # Back Propagation
            dL_dW1, dL_dW2= self.back_propagation(
                x, y, z1, a1, z2, a2, W1, W2
            )

            # Update Parameters
            W1, W2 = self.update_parameters(
                W1, W2, dL_dW1, dL_dW2, 0.01
            )

            if epoch % 1 == 0:  # After every 50th training, show the models accuracy
                # print(f"Training Iteration/Epoch:\n{epoch}\n")
                # print(f"Accuracy per Label:\n{self.calculate_accuracy_per_label(a2, y)}\n")
                # print(f"Total Model Accuracy:\n{self.calculate_accuracy(a2, y) * 100}%\n")
                print(
                    f"Predictions:\n{np.argmax(a2, axis=0)[:10]}\n"
                    f"True labels:\n{y[:10]}\n"
                    #   f"Parameters:\n "
                    #   f"W1:\n{W1}\n"
                      # f"W2:\n{W2}\n"
                      # f"Layers:\n"
                      # f"Z1:\n{np.argwhere(np.isnan(z1))}\n"
                      # f"Z2:\n{np.argwhere(np.isnan(z2))}\n"
                      # f"Softmax pdf iteration {epoch}: {np.argwhere(np.isnan(a2))}"
                      # f"Paramters:\nW1: {W1}\nW2: {W2}\nb1: {b1}\nb2: {b2}"
                      # f"Derivatives:\ndL/dW1: {dL_dW1}\ndL/dW2: {dL_dW2}\ndL/db1: {dL_db1}\ndL/db2: {dL_db2}"
                      )

