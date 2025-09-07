import numpy as np
import os
from sklearn.model_selection import train_test_split
from models.LinearRegression import LinearRegression
from sklearn import linear_model

WINE_FILE_PATH = r"data\wine.txt"

def import_wine(
    filepath: str, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper function to import the wine dataset.

    Parameters
    ----------
    filepath : str
        Path to wine.txt.
    test_size : float
        The fraction of the dataset set aside for testing

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A 4-tuple (X_train, X_test, Y_train, Y_test) that gives the testing and training data.
    """

    # Check if the file exists
    if not os.path.exists(filepath):
        print(f"The file {filepath} does not exist")
        exit()

    # Load in the dataset
    data = np.loadtxt(filepath, skiprows=1)
    X, Y = data[:, 1:], data[:, 0]

    # Normalize the inputs
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    return X_train, X_test, Y_train, Y_test


def test_linreg() -> None:
    '''
    Compares the Linear Regression model implemented in LinearRegression.py
    with the Linear Regression model implemented in sklearn. Uses the wine dataset
    to attempt to predict the quality of wine based on its attributes.
    '''
    X_train, X_test, Y_train, Y_test = import_wine(WINE_FILE_PATH)

    num_features = X_train.shape[1]

    # Padding the inputs with a bias
    X_train_b = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    X_test_b = np.append(X_test, np.ones((len(X_test), 1)), axis=1)

    #### Testing Personal Model With Scikit Learn Model ######
    print("---- LINEAR REGRESSION w/ Matrix Inversion ---")
    solver_model = LinearRegression(num_features)
    solver_model.train(X_train_b, Y_train)

    assert solver_model.average_loss(X_train_b, Y_train) < 1.0
    assert solver_model.average_loss(X_test_b, Y_test) < 1.0

    linear_model_sklearn = linear_model.LinearRegression(fit_intercept=False)
    linear_model_sklearn.fit(X_train_b, Y_train) # Remove bias column for sklearn
    # assert linear_model_sklearn.score(X_train_b, Y_train) == solver_model.average_loss(X_train_b, Y_train)
    # assert linear_model_sklearn.score(X_test, Y_test) == solver_model.average_loss(X_test_b, Y_test)
    np.testing.assert_allclose(
        solver_model.predict(X_test_b),  # your model uses bias column
        linear_model_sklearn.predict(X_test_b),              # sklearn adds its own intercept
        atol=1e-8,
    )