import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB as SkBernoulliNB
from models.NaiveBayes import NaiveBayes

CREDIT_FILE_PATH = r"data/german_numerical-binsensitive.csv"

def get_credit() -> tuple[np.ndarray, ...]:
    """
    Reads and preprocesses German Credit dataset

    Parameters
    ----------
    None

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - X_train: Training input data
        - X_val: Validation input data
        - y_train: Training output data
        - y_val: Validation outpupt data
    """
    np.random.seed(0)

    data = pd.read_csv(CREDIT_FILE_PATH) 

    # MONTH categorizing
    data["month"] = pd.cut(
        data["month"], 3, labels=["month_1", "month_2", "month_3"], retbins=True
    )[0]
    # month bins: [ 3.932     , 26.66666667, 49.33333333, 72.        ]
    a = pd.get_dummies(data["month"])
    data = pd.concat([data, a], axis=1)
    data = data.drop(["month"], axis=1)

    # CREDIT categorizing
    data["credit_amount"] = pd.cut(
        data["credit_amount"],
        3,
        labels=["cred_amt_1", "cred_amt_2", "cred_amt_3"],
        retbins=True,
    )[0]
    # credit bins: [  231.826,  6308.   , 12366.   , 18424.   ]
    a = pd.get_dummies(data["credit_amount"])
    data = pd.concat([data, a], axis=1)
    data = data.drop(["credit_amount"], axis=1)

    for header in [
        "investment_as_income_percentage",
        "residence_since",
        "number_of_credits",
    ]:
        a = pd.get_dummies(data[header], prefix=header)
        data = pd.concat([data, a], axis=1)
        data = data.drop([header], axis=1)

    # change from 1-2 classes to 0-1 classes
    data["people_liable_for"] = data["people_liable_for"] - 1
    data["credit"] = (
        -1 * (data["credit"]) + 2
    )  # original encoding 1: good, 2: bad. we switch to 1: good, 0: bad

    # balance dataset
    data = data.reindex(np.random.permutation(data.index))  # shuffle
    pos = data.loc[data["credit"] == 1]
    neg = data.loc[data["credit"] == 0][:350]
    combined = pd.concat([pos, neg])

    y = data.iloc[:, data.columns == "credit"].to_numpy()
    x = data.drop(["credit", "sex", "age", "sex-age"], axis=1).to_numpy()

    # split into train and validation
    X_train, X_val, y_train, y_val = (
        x[:350, :],
        x[351:526, :],
        y[:350, :].reshape(
            [
                350,
            ]
        ),
        y[351:526, :].reshape(
            [
                175,
            ]
        ),
    )

    return X_train, X_val, y_train, y_val


def test_naive_bayes() -> None:
    """
    Driving function
    """
    np.random.seed(0)

    # Get the dataset with valid inputs for the Naive Bayes model
    X_train, X_val, y_train, y_val = get_credit()

    # My Implementation of Naive Bayes
    my_model = NaiveBayes(2)
    my_model.train(X_train, y_train)
    my_acc = my_model.accuracy(X_val, y_val)

    # Sklearn's Implementation of Bernoulli Naive Bayes
    ref = SkBernoulliNB(alpha=1.0, binarize=None, fit_prior=True)
    ref.fit(X_train, y_train)
    acc_ref = ref.score(X_val, y_val)

    # Guarentee that my implementation has a reasonable accuracy
    assert acc_ref > 0.7
    # Guarentee that my implementation is close to sklearn's accuracy
    assert np.isclose(my_acc, acc_ref, atol=0.1)