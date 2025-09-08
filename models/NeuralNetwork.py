from typing import Callable
import numpy as np


def l2_loss(predictions: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes L2 loss (sum squared loss) between predictions and Y.

    Parameters
    ----------
    predictions : np.ndarray
        A 1D Numpy array.
    Y : np.ndarray
        A 1D Numpy array of the same size as predictions.

    Returns
    -------
    float
        L2 loss using between predictions and Y.
    """
    return sum(np.square(predictions - Y))


def sigmoid(a: np.ndarray) -> np.ndarray:
    """
    Sigmoid function, given by sigma(a) =  1/(1 + exp(-a)), applied component-wise

    Parameters
    ----------
    a : np.ndarray
        A Numpy array.

    Returns
    -------
    np.ndarray
        Sigmoid function evaluation on a.
    """
    return np.where(a > 0, 1 / (1 + np.exp(-a)), np.exp(a) / (np.exp(a) + np.exp(0)))


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    """
    First derivative of the sigmoid function with respect to a.

    Parameters
    ----------
    a : np.ndarray
        A Numpy array.

    Returns
    -------
    np.ndarray
        Sigmoid derivative evaluation on a.
    """
    s = sigmoid(a)
    return s * (1-s)

def ReLU(a: np.ndarray) -> np.ndarray:
    return np.where(a > 0, a, 0)

def ReLU_derivative(a: np.ndarray) -> np.ndarray:
    return np.where(a > 0, 1, 0)


class OneLayerNN:
    """
    Constructor for a one layer neural network.

    Attributes
    ----------
    w : np.ndarray
        The weights of the first layer of the neural network model.
    learning_rate : float
        The learning rate to use for SGD.
    epochs : int
        The number of times to pass through the dataset.
    o : np.ndarray
        The output of the network.
    """

    def __init__(self):
        """
        Initializes a one layer neural network.
        """
        # initialize self.w in train()
        self.w = None
        self.learning_rate = 0.001
        self.epochs = 25
        self.o = None

    def train(self, X: np.ndarray, Y: np.ndarray, print_loss: bool = True) -> None:
        """
        Training loop with SGD for OneLayerNN model.

        Parameters
        ----------
        X : np.ndarray
            2D Numpy array, each row representing one example.
        Y : np.ndarray
            1D Numpy array, each entry a label for the corresponding row (example) in X.
        print_loss : bool, default=True
            If True, print the loss after each epoch.

        Returns
        -------
        None
        """
        # Initialize self.w. In the OneLayerNN, every example
        # has a 1 as the last feature, so no separate bias term is needed.
        self.w = np.random.uniform(0,1, X.shape[1])

        # Train network for certain number of epochs defined in self.epochs
        for epoch in range(self.epochs):
            # Shuffle the examples (X) and labels (Y)
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_random, Y_random = X[indices], Y[indices]

            # Iterate over each example for each epoch
            for i, example in enumerate(X_random): 
                # Perform the forward and backward pass on the current example
                self.forward_pass(example) 
                self.backward_pass(example, Y_random[i])


            # Print the loss after every epoch
            if print_loss:
                print("Epoch: {} | Loss: {}".format(epoch, self.loss(X, Y)))


    def forward_pass(self, x: np.ndarray) -> None:
        """
        Computes the output of the network given an input vector x and stores the result in self.o.

        Parameters
        ----------
        x : np.ndarray
            1D Numpy array, representing one example.

        Returns
        -------
        None
        """
        # Calculate output of neural network on x
        self.o = np.dot(x, self.w)


    def backward_pass(self, x: np.ndarray, y: float) -> None:
        """
        First computes the gradient of the loss on an example with respect to self.w.
        Then updates self.w. Should only be called after self.forward_pass.

        Parameters
        ----------
        x : np.ndarray
            1D Numpy array, representing one example.
        y : float
            Scalar, the label for the example x.

        Returns
        -------
        None
        """
        # Calculate the gradient of the weights
        gradient = 2 * (self.o - y) * x

        # Update the weights using the gradient
        self.w -= gradient * self.learning_rate

    def loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Returns the total squared error on some dataset (X, Y).

        Parameters
        ----------
        X : np.ndarray
            2D Numpy array, each row representing one example.
        Y : np.ndarray
            1D Numpy array, each entry a label for the corresponding row (example) in X.

        Returns
        -------
        float
            A float which is the squared error of the model on the dataset.
        """
        # Perform the forward pass and compute the l2 loss
        outputs = np.zeros([X.shape[0]])
        for i in range(X.shape[0]):
            self.forward_pass(X[i])
            outputs[i] = self.o

        return l2_loss(outputs, Y)

    def average_loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Returns the mean squared error on some dataset (X, Y).
        MSE = Total squared error / # of examples

        Parameters
        ----------
        X : np.ndarray
            2D Numpy array, each row representing one example.
        Y : np.ndarray
            1D Numpy array, each entry a label for the corresponding row (example) in X.

        Returns
        -------
        float
            A float which is the mean squared error of the model on the dataset.
        """
        return self.loss(X, Y) / X.shape[0]


class TwoLayerNN:
    """
    Two layer neural network trained with Stochastic Gradient Descent (SGD).

    Attributes
    ----------
    activation : Callable
        The activation function applied after the first layer.
    activation_derivative : Callable
        The derivative of the activation function. Used for training.
    hidden_size : int
        The hidden size of the network.
    learning_rate : float
        The learning rate to use for SGD.
    epochs : int
        The number of times to pass through the dataset.

    Other variable naming conventions:
        Letters:
            w = weights layers
            b = bias layers
            a = output of the first layer computed during forward pass
            o = the activated output of the first layer computed during forward pass
                (ie, result of applying activation function to an a)

    Numbers serve as layer indices. For example, w12 would be the weights between
    the first and second layer.
    """

    def __init__(
        self,
        hidden_size: int,
        activation: Callable = sigmoid,
        activation_derivative: Callable = sigmoid_derivative,
    ):
        """
        Constructor for a two layer neural network.

        Parameters
        ----------
        hidden_size : int
            The hidden size of the network.
        activation : Callable, default=sigmoid
            The activation function applied after the first layer.
        activation_derivative : Callable, default=sigmoid_derivative
            The derivative of the activation function. Used for training.
        """
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.hidden_size = hidden_size
        self.learning_rate = 0.01
        self.epochs = 25

        # initialize the following weights and biases in the train() method
        self.W01 = None
        self.b1 = None
        self.W12 = None
        self.b2 = None

        # Outputs of each layer
        self.a1 = None
        self.o1 = None
        self.a2 = None
        self.o2 = None

    def train(self, X: np.ndarray, Y: np.ndarray, print_loss: bool = True) -> None:
        """
        Trains the TwoLayerNN with SGD using Backpropagation.

        Parameters
        ----------
        X : np.ndarray
            2D Numpy array, each row representing one example.
        Y : np.ndarray
            1D Numpy array, each entry a label for the corresponding row (example) in X.
        print_loss : bool, default=True
            If True, print the loss after each epoch.

        Returns
        -------
        None
        """
        # Weight and bias initialization
        # layer 1 weights (W01): hidden_size x n_features (2D matrix)
        self.W01 = np.random.uniform(0,1, size=(self.hidden_size, X.shape[1]))
        # layer 1 bias (b1): hidden_size (1D vector)
        self.b1 = np.random.uniform(0,1,size=self.hidden_size)
        # layer 2 weights (W12): 1 x hidden_size (2D matrix)
        self.W12 = np.random.uniform(0,1,size=(1, self.hidden_size))
        # layer 2 bias (b2): 1 (1D vector)
        self.b2 = np.random.uniform(0,1)

        # Train network for certain number of epochs
        for epoch in range(self.epochs):
            # Shuffle the examples (X) and labels (Y)
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_random, Y_random = X[indices], Y[indices]

            # Iterate over each example for each epoch
            for i, example in enumerate(X_random): 
                # Perform the forward and backward pass on the current example
                self.forward_pass(example) 
                self.backward_pass(example, Y_random[i])

            # Print the loss after every epoch
            if print_loss:
                print("Epoch: {} | Loss: {}".format(epoch, self.loss(X, Y)))
        

    def forward_pass(self, x: np.ndarray) -> None:
        """
        Computes the outputs of the network given an input vector x. Stores the activation function input
        (weighted sum for each neuron) for each layer in self.a* and the post-activation output for each
        layer in self.o*.

        Parameters
        ----------
        x : np.ndarray
            1D Numpy array, representing one example.

        Returns
        -------
        None
        """
        # Calculate output of neural network on X
        self.a1 = self.W01 @ x + self.b1
        self.o1 = self.activation(self.a1)
        self.a2 = self.W12 @ self.o1 + self.b2
        self.o2 = self.a2


    def backprop(
        self, x: np.ndarray, y: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the gradients of the loss w.r.t the weights and biases of each layer
        using the backpropagation algorithm.

        Parameters
        ----------
        x : np.ndarray
            1D Numpy array, representing one example.
        y : float
            Scalar, the label for the example x.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (W12_grad, b2_grad, W01_grad, b1_grad)
        """
        # 2nd Hidden Layer 
        b2_grad = 2 * (self.o2 - y)
        W12_grad = np.outer(b2_grad, self.o1)
        
        # Middle Step
        a1_grad = self.activation_derivative(self.a1)

        # 1st Hidden Layer
        b1_grad = np.matmul(self.W12.T, b2_grad) * a1_grad
        W01_grad = np.outer(b1_grad, x)

        return (W12_grad, b2_grad, W01_grad, b1_grad) 

    def backward_pass(self, x: np.ndarray, y: float) -> None:
        """
        First computes the gradient of the loss on an example with respect to self.W01, self.b1, self.W12,
        and self.b2 by calling self.backprop. Then updates all those parameters. Should only be called
        after self.forward_pass.

        Parameters
        ----------
        x : np.ndarray
            1D Numpy array, representing one example.
        y : float
            Scalar, the label for the example x.

        Returns
        -------
        None
        """
        # Compute the gradients for the model's weights by calling self.backprop
        W12_grad, b2_grad, W01_grad, b1_grad = self.backprop(x,y)

        # Update the weights using gradient descent
        self.W12 -= W12_grad * self.learning_rate
        self.b2 -= b2_grad * self.learning_rate
        self.W01 -= W01_grad * self.learning_rate
        self.b1 -= b1_grad * self.learning_rate

    def loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Returns the total squared error on some dataset (X, Y).

        Parameters
        ----------
        X : np.ndarray
            2D Numpy array, each row representing one example.
        Y : np.ndarray
            1D Numpy array, each entry a label for the corresponding row (example) in X.

        Returns
        -------
        float
            A float which is the squared error of the model on the dataset.
        """
        # Perform the forward pass and compute the l2 loss
        outputs = np.zeros([X.shape[0]])
        for i in range(X.shape[0]):
            self.forward_pass(X[i])
            outputs[i] = self.o2

        return l2_loss(outputs, Y)

    def average_loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Returns the mean squared error on some dataset (X, Y).
        MSE = (Total squared error) / (# of examples)

        Parameters
        ----------
        X : np.ndarray
            2D Numpy array, each row representing one example.
        Y : np.ndarray
            1D Numpy array, each entry a label for the corresponding row (example) in X.

        Returns
        -------
        float
            A float which is the mean squared error of the model on the dataset.
        """
        return self.loss(X, Y) / X.shape[0]
