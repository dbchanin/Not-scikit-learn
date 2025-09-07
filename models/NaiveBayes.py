import numpy as np


class NaiveBayes(object):
    """Bernoulli Naive Bayes model

    Parameters
    ----------
    n_classes : int
        The number of classes.

    Attributes
    ----------
    n_classes: int
        The number of classes.
    attr_dist: np.ndarray
        2D (n_classes x n_attributes) NumPy array of the attribute distributions
    label_priors: np.nparray
        1D NumPy array of the priors distribution
    """

    def __init__(self, n_classes: int) -> None:
        """
        Constructor for NaiveBayes model with n_classes.
        """
        self.n_classes = n_classes
        self.attr_dist = None
        self.label_priors = None

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Trains the model using maximum likelihood estimation.

        Parameters
        ----------
        X_train: np.ndarray
            a 2D (n_examples x n_attributes) numpy array
        y_train: np.ndarray
            a 1D (n_examples) numpy array

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple consisting of:
                1) a 2D numpy array of the attribute distributions
                2) a 1D numpy array of the priors distribution
        """
        n_examples = X_train.shape[0]
        n_attributes = X_train.shape[1]

        self.label_priors = np.zeros(self.n_classes, dtype='float64')
        for i in range(self.n_classes):
            self.label_priors[i] = np.sum(y_train == i) 

        # Leplace Smoth self.label_priors using class counts
        self.label_priors += 1
        self.label_priors /= (n_examples + self.n_classes)
        
        # Get the data within each class
        data = {}
        for i in range(self.n_classes):
            data[i] = X_train[y_train == i]

        # Get the attribute labels tabel
        self.attr_dist = np.zeros((self.n_classes, n_attributes), dtype='float64')
        for i in range(self.n_classes):
            # Get the count for each feature
            data_in_class = data[i]
            self.attr_dist[i] = np.sum(data_in_class, 0) 

            # Leplace Smooth
            self.attr_dist[i] += 1
            self.attr_dist[i] /= len(data[i]) + 2

        return (self.attr_dist, self.label_priors)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Outputs a predicted label for each input in inputs.

        Parameters
        ----------
        inputs: np.ndarray
            a 2D NumPy array containing inputs

        Returns
        -------
        np.ndarray
            a 1D numpy array of predictions
        """
        predictions = np.zeros(inputs.shape[0])
        # Calculate probability for each input
        for input_index, input in enumerate(inputs):

            class_probabilities = np.zeros(self.n_classes)
            for class_index in range(self.n_classes):
                # Get the adjusted probabilities for the input set                
                adjusted_probs = np.where(input == 0, 1 - self.attr_dist[class_index, :], self.attr_dist[class_index, :])

                # get the probability for this class
                class_probabilities[class_index] = self.label_priors[class_index] * np.prod(adjusted_probs)

            # Predict the most likely class
            predictions[input_index] = np.argmax(class_probabilities)

        return predictions
                    

    def accuracy(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Outputs the accuracy of the trained model on a given dataset (data).

        Parameters
        ----------
        X_test: np.ndarray
            a 2D numpy array of examples
        y_test: np.ndarray
            a 1D numpy array of labels

        Returns
        -------
        float
            a float number indicating accuracy (between 0 and 1)
        """
        predictions = self.predict(X_test)
        
        sum_correct = np.sum(predictions == y_test)

        return sum_correct / len(predictions)