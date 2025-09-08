from utils import Visualizer
import matplotlib.axes as Axes
import numpy as np
from sklearn import svm


def seperate_support_vectors(
    arr: np.ndarray, sv_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Seperates a 2d numpy array into 2 2d arrays. One array will contain only the rows corresponding
    to the given indices. The other will only contain the remaining rows.

    Parameters
    ----------
    arr: np.ndarray
        The 2d numpy array to be separated
    sv_idx: np.ndarray
        The 1d array of indices indicating which rows to remove from arr and place into a new array

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two numpy arrays with the same number of columns as arr.
    """
    support_vectors = arr[sv_idx]
    non_support_vectors = np.delete(arr, sv_idx, axis=0)

    return support_vectors, non_support_vectors


class SupportVectorVisualization(Visualizer):
    def __init__(self, svm: svm):
        self.svm = svm

    def generate_x2_vals(
        self, x1_axis_points: np.ndarray, coefficients: np.ndarray, intercept: float
    ) -> np.ndarray:
        """
        Calculates the values of x_2 that lie on H(x) = 0 corresponding to each inputted value of x_1.

        Parameters
        ----------
        x1_axis_points: np.ndarray
            A numpy vector of x_1 values.
        coefficients: np.ndarray
            A numpy vector of coefficients, [w_1, w_2].
        intercept: float
            A scalar value representing the bias.

        Returns
        -------
        np.ndarray
            A numpy vector of x_2 values
        """
        w1, w2 = coefficients[0], coefficients[1]
        slope = -w1/w2
        x2_vals = (slope * x1_axis_points) - (intercept/w2) 

        return x2_vals

    def generate_margin_points(
        self, x2_vals: np.ndarray, coefficients: np.ndarray
    ) -> np.ndarray:
        """
        For each point of the decision boundary, calculates the point on the margin
        below the boundary and above the boundary.

        Parameters
        ----------
        x2_vals: np.ndarray
            A numpy vector of x_2 values
        coefficients: np.ndarray
            A numpy vector of coefficients, [w_1, w_2]

        Returns
        -------
        np.ndarray
            A numpy vector representing the upper margin points and a numpy vector representing the lower margin points
        """
        # Get the vertical distance from the hyperplane to the margin
        w1, w2 = coefficients[0], coefficients[1]
        slope = -w1/w2
        vertical_distance = np.sqrt(1 + slope**2) / np.linalg.norm(coefficients)
        upper_margin_vals, lower_margin_vals = x2_vals + vertical_distance, x2_vals - vertical_distance

        return upper_margin_vals, lower_margin_vals

    def visualize_solution(
        self, X: np.ndarray, Y: np.ndarray, x1_axis_points: np.ndarray, ax: Axes
    ) -> None:
        """
        Modifies an axes object by plotting the margins, and the data points in R2. Outlines the support vectors.

        Parameters
        ----------
        X: np.ndarray
            A n_training_samples x n_features numpy matrix of training data
        Y: np.ndarray
            A numpy vector of labels for X
        x1_axis_points: np.ndarray
            A numpy vector of x_1 values
        ax: Axes
            An axes object

        Returns
        -------
        None
        """
        # These variables have n_classes rows, but since we only have 1 class
        # We flatten them into vectors
        coefficients = self.svm.coef_.flatten()
        intercept = self.svm.intercept_.flatten()

        # Get the x2 and margin values
        x2_vals = self.generate_x2_vals(x1_axis_points, coefficients, intercept)
        upper_margin_vals, lower_margin_vals = self.generate_margin_points(x2_vals, coefficients)
        # Seperate the support vector points from the non support vector points
        support_vectors, non_support_vectors = seperate_support_vectors(X, self.svm.support_)
        support_vector_labels, non_support_vector_labels = seperate_support_vectors(Y, self.svm.support_)

        ax.plot(x1_axis_points, x2_vals)  # plots (x1, x2)
        ax.plot(x1_axis_points, upper_margin_vals, "k--")  # plots (x1, upper margin)
        ax.plot(x1_axis_points, lower_margin_vals, "k--")  # plots (x1, lower margin)
        ax.scatter(
            support_vectors[:, 0],
            support_vectors[:, 1],
            s=100,
            c=support_vector_labels,
            edgecolors="red",
        )
        ax.scatter(
            non_support_vectors[:, 0],
            non_support_vectors[:, 1],
            c=non_support_vector_labels,
        )

        ax.set_ylabel(r"$x_2$")
        ax.set_xlabel(r"$x_1$")
        ax.label_outer()
