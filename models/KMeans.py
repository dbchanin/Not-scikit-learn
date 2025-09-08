import numpy as np
from random import sample


def init_centroids(k: int, inputs: np.ndarray) -> np.ndarray:
    """
    Selects k random rows from inputs and returns them as the chosen centroids.

    Parameters
    ----------
    k : int
        Number of cluster centroids.
    inputs : np.ndarray
        A 2D Numpy array, each row of which is one input.

    Returns
    -------
    np.ndarray
        A Numpy array of k cluster centroids, one per row.
    """
    # unique_rows = np.unique(inputs, axis=0)
    return np.array(sample(list(inputs), k=k))

def euclidean_dist(x, y):
    return np.linalg.norm(x - y, axis=-1)

def assign_step(inputs: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Determines a centroid index for every row of the inputs using Euclidean Distance.

    Parameters
    ----------
    inputs : np.ndarray
        Inputs of data, a 2D Numpy array.
    centroids : np.ndarray
        A Numpy array of k current centroids.

    Returns
    -------
    np.ndarray
        A Numpy array of centroid indices, one for each row of the inputs.
    """
    result = np.zeros(inputs.shape[0], dtype=int)
    for i, data_point in enumerate(inputs):
        result[i] = np.argmin(euclidean_dist(centroids, data_point))

    return result


def update_step(inputs: np.ndarray, indices: np.ndarray, k: int) -> np.ndarray:
    """
    Computes the centroid for each cluster.

    Parameters
    ----------
    inputs : np.ndarray
        Inputs of data, a 2D Numpy array.
    indices : np.ndarray
        A Numpy array of centroid indices, one for each row of the inputs.
    k : int
        Number of cluster centroids.

    Returns
    -------
    np.ndarray
        A Numpy array of k cluster centroids, one per row.
    """
    new_centroids = np.zeros((k, inputs.shape[1]))
    counts = np.zeros(k)
    for i, row in enumerate(inputs):
        current_centroid = indices[i]
        new_centroids[current_centroid] += row
        counts[current_centroid] += 1

    # Get the average distance from each centroid
    for i in range(k):
        count = counts[i]
        if count > 0:
            new_centroids[i] /= count
        else:
            new_centroids[i] = inputs[np.random.randint(len(inputs))]

    return new_centroids

def isTolerant(curr_centroids: np.ndarray, updated_centroids: np.ndarray, tolerance: float) -> bool:
    relative_change = np.linalg.norm(updated_centroids - curr_centroids) / np.linalg.norm(curr_centroids)
    return relative_change < tolerance

def kmeans(inputs: np.ndarray, k: int, max_iter: int, tol: float) -> np.ndarray:
    """
    Runs the K-means algorithm on n rows of inputs using k clusters via iterative improvement.

    Parameters
    ----------
    inputs : np.ndarray
        Inputs of data, a 2D Numpy array.
    k : int
        Number of cluster centroids.
    max_iter : int
        The maximum number of times the algorithm can iterate trying to optimize the centroid values.
    tol : float
        The tolerance we determine convergence with when compared to the ratio as stated on handout.

    Returns
    -------
    np.ndarray
        A Numpy array of k cluster centroids, one per row.
    """
    curr_centroids = init_centroids(k, inputs)
    for _ in range(max_iter):
        centroid_indices = assign_step(inputs, curr_centroids)
        updated_centroids = update_step(inputs, centroid_indices, k)

        if isTolerant(curr_centroids, updated_centroids, tol): 
            return updated_centroids
        else:
            curr_centroids = updated_centroids

    return curr_centroids


class KmeansClassifier(object):
    """
    K-Means Classifier via Iterative Improvement.

    Attributes
    ----------
    k : int, default=10
        The number of clusters to form as well as the number of centroids to
        generate.
    tol : float, default=1e-6
        Value specifying our convergence criterion. If the ratio of the
        distance each centroid moves to the previous position of the centroid
        is less than this value, then we declare convergence.
    max_iter : int, default=500
        The maximum number of times the algorithm can iterate trying to optimize the centroid values,
        the default value is set to 500 iterations.
    cluster_centers : np.ndarray
        A Numpy array where each element is one of the k cluster centers.
    """

    def __init__(
        self, n_clusters: int = 10, max_iter: int = 500, threshold: float = 1e-6
    ):
        """
        Initiate K-Means with some parameters

        Parameters
        ----------
        n_clusters : int, default=10
            The number of clusters to form as well as the number of centroids to
            generate.
        max_iter : int, default=500
            The maximum number of times the algorithm can iterate trying to optimize the centroid values,
            the default value is set to 500 iterations.
        threshold : float, default=1e-6
            Value specifying our convergence criterion. If the ratio of the
            distance each centroid moves to the previous position of the centroid
            is less than this value, then we declare convergence.
        """
        self.k = n_clusters
        self.tol = threshold
        self.max_iter = max_iter
        self.cluster_centers = np.array([])

    def train(self, X: np.ndarray) -> None:
        """
        Compute K-Means clustering on each class label and store your result in self.cluster_centers.

        Parameters
        ----------
        X : np.ndarray
            inputs of training data, a 2D Numpy array

        Returns
        -------
        None
        """
        self.cluster_centers = kmeans(X, self.k, self.max_iter, self.tol)

    def predict(self, X: np.ndarray, centroid_assignments: list[int]) -> np.ndarray:
        """
        Predicts the label of each sample in X based on the assigned centroid_assignments.

        Parameters
        ----------
        X : np.ndarray
            A dataset as a 2D Numpy array.
        centroid_assignments : list[int]
            A Python list of 10 digits (0-9) representing the interpretations of the digits of the plotted centroids.

        Returns
        -------
        np.ndarray
            A Numpy array of predicted labels.
        """
        predictions = np.zeros(shape=(X.shape[0]))
        for i, data_point in enumerate(X):
            centroid_num = np.argmin(euclidean_dist(data_point, self.cluster_centers))
            predictions[i] = centroid_assignments[centroid_num]

        return predictions

    def accuracy(
        self,
        data: tuple[list[np.ndarray], list[np.ndarray]],
        centroid_assignments: list[int],
    ) -> float:
        """
        Compute accuracy of the model when applied to data.

        Parameters
        ----------
        data : tuple[list[np.ndarray], list[np.ndarray]]
            A namedtuple including inputs and labels.
        centroid_assignments : list[int]
            A python list of 10 digits (0-9) representing your interpretations of the digits of the plotted centroids from plot_Kmeans (in order from left ot right).

        Returns
        -------
        float
            A float number indicating accuracy.
        """
        pred = self.predict(data.inputs, centroid_assignments)
        return np.mean(pred == data.labels)