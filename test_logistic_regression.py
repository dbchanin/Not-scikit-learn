import numpy as np
from sklearn.model_selection import train_test_split
from models.LogisticRegression import LogisticRegression as MyLogisticRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

CENSUS_FILE_PATH = r"data/normalized_census.csv"
NUM_CLASSES = 3
BATCH_SIZE = 75
CONV_THRESHOLD = 1e-5

def import_census(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper function to import the census dataset. Uses sklearn's test/train split.

    Parameters
    ----------
    filepath : str
        Path to census dataset

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A 4-tuple (X_train, X_test, Y_train, Y_test) that gives the testing and training data.
    """
    data = np.genfromtxt(filepath, delimiter=",", skip_header=False)
    X = data[:, :-1]
    Y = data[:, -1].astype(int)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0
    )
    return X_train, Y_train, X_test, Y_test


def test_logist_regression() -> float:
    """Runs the model training and test loop on the 1994 census dataset and compares
    the accuracy of my implementation with sklearn's implementation. The model predicts
    the education of a person based on certain census attributes.

    Returns
    -------
    float
        Returns model accuracy
    """
    X_train, Y_train, X_test, Y_test = import_census(CENSUS_FILE_PATH)
    num_features = X_train.shape[1]

    # Add a bias
    X_train_b = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    X_test_b = np.append(X_test, np.ones((len(X_test), 1)), axis=1)

    # My Implementation of Logistic Regression
    my_model = MyLogisticRegression(num_features, NUM_CLASSES, BATCH_SIZE, CONV_THRESHOLD)
    my_model.train(X_train_b, Y_train)
    my_acc = my_model.accuracy(X_test_b, Y_test)

    # Sklearn's Implementation of Logistic Regression
    # use stochastic gradient descent as the solver 
    ref = SklearnLogisticRegression(solver="saga", max_iter=1000)
    ref.fit(X_train, Y_train)
    acc_ref = ref.score(X_test, Y_test)

    # Ensure that my model is an accurate predictor
    assert my_acc >= 0.85
    # Ensure that my model is comparable to sklearn's implementation
    assert (abs(my_acc - acc_ref) < 0.1 or my_acc > acc_ref)