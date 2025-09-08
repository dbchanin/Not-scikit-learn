# tests/test_neural_networks.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from models.NeuralNetwork import OneLayerNN, TwoLayerNN, ReLU, ReLU_derivative
import torch
import torch.nn as nn

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
        (X_train, X_test, Y_train, Y_test)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist")

    data = np.loadtxt(filepath, skiprows=1)
    X, Y = data[:, 1:], data[:, 0]

    # Normalize features only
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
    return X_train, X_test, Y_train, Y_test


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def test_neural_networks_on_wine() -> None:
    """
    Trains OneLayerNN (with bias column) and TwoLayerNN (no bias column),
    asserts sensible performance vs. a naive baseline and (for OneLayerNN)
    rough parity with sklearn's LinearRegression.
    """
    np.random.seed(0)

    X_train, X_test, Y_train, Y_test = import_wine(WINE_FILE_PATH)
    n_features = X_train.shape[1]

    # Baseline (predict train mean)
    baseline_pred = np.full_like(Y_test, fill_value=Y_train.mean(), dtype=float)
    baseline_mse = _mse(Y_test, baseline_pred)

    # ----- 1-Layer NN -----
    X_train_b = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    X_test_b = np.append(X_test, np.ones((len(X_test), 1)), axis=1)

    one = OneLayerNN()
    one.train(X_train_b, Y_train, print_loss=False)
    one_train_mse = one.average_loss(X_train_b, Y_train)
    one_test_mse = one.average_loss(X_test_b, Y_test)

    # Should beat naive baseline by a comfortable margin
    assert one_test_mse <= baseline_mse * 0.9

    # Compare to sklearn LinearRegression (closed-form)
    sk = linear_model.LinearRegression(fit_intercept=False)
    sk.fit(X_train_b, Y_train)
    sk_mse_test = _mse(Y_test, sk.predict(X_test_b))

    assert one_test_mse <= sk_mse_test * 2.0

    # ----- 2-Layer NN (sigmoid) -----
    two = TwoLayerNN(hidden_size=10)
    two.train(X_train, Y_train, print_loss=False)
    two_train_mse = two.average_loss(X_train, Y_train)
    two_test_mse = two.average_loss(X_test, Y_test)

    # Also should beat naive baseline
    assert two_test_mse <= baseline_mse * 0.9

    two_relu = TwoLayerNN(hidden_size=10, activation=ReLU, activation_derivative=ReLU_derivative)
    two_relu.train(X_train, Y_train, print_loss=False)
    two_relu_test_mse = two_relu.average_loss(X_test, Y_test)

    # ReLU variant should also beat baseline
    assert two_relu_test_mse <= baseline_mse * 0.9

    # Basic sanity: all losses are finite
    for v in [one_train_mse, one_test_mse, two_train_mse, two_test_mse, two_relu_test_mse]:
        assert np.isfinite(v)


def test_two_layer_matches_pytorch() -> None:
    """
    Compare my TwoLayerNN implementation against a PyTorch MLP of the same structure
    """
    np.random.seed(0)
    torch.manual_seed(0)

    X_train, X_test, Y_train, Y_test = import_wine(WINE_FILE_PATH)
    hidden = 10

    # My implementation ----
    mine = TwoLayerNN(hidden_size=hidden)  # defaults: sigmoid + linear output, lr=0.01, epochs=25
    mine.train(X_train, Y_train, print_loss=False)
    my_test_mse = mine.average_loss(X_test, Y_test)

    # PyTorch reference (same structure & SGD loop) ----
    class TorchTwoLayer(nn.Module):
        def __init__(self, in_features: int, hidden_size: int):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden_size)
            self.act = nn.Sigmoid()
            self.fc2 = nn.Linear(hidden_size, 1)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x))).squeeze(-1)

    # Build model
    net = TorchTwoLayer(in_features=X_train.shape[1], hidden_size=hidden)
    with torch.no_grad():
        for p in net.parameters():
            p.uniform_(0.0, 1.0)

    optim = torch.optim.SGD(net.parameters(), lr=0.01)
    for _ in range(25):
        idx = np.arange(len(X_train))
        np.random.shuffle(idx)
        for i in idx:
            x = torch.from_numpy(X_train[i]).float()
            y = torch.tensor(Y_train[i], dtype=torch.float32)

            optim.zero_grad()
            pred = net(x)
            loss = (pred - y) ** 2  # single-sample squared error
            loss.backward()
            optim.step()

    # Evaluate PyTorch model
    with torch.no_grad():
        Xt = torch.from_numpy(X_test).float()
        preds = net(Xt).cpu().numpy()
    torch_test_mse = float(np.mean((preds - Y_test) ** 2))

    # Parity assertion: custom NN should be within a reasonable factor of torch MLP
    assert my_test_mse <= torch_test_mse * 2.0, (
        f"Custom TwoLayerNN MSE ({my_test_mse:.4f}) not in the same ballpark as "
        f"PyTorch MLP ({torch_test_mse:.4f})."
    )

    # Basic sanity
    assert np.isfinite(my_test_mse)
    assert np.isfinite(torch_test_mse)

