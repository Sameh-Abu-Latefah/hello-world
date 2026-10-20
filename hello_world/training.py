import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix



def train_decision_tree(X_train, y_train, max_depth: int = 3) -> DecisionTreeClassifier:
    """Train a decision tree on train data and return the fitted model."""
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Train a logistic regression on train data and return the fitted model."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators: int = 100,
                        max_depth: int = 3) -> RandomForestClassifier:
    """Train a random forest on train data and return the fitted model."""
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    """Accuracy on train, accuracy on test, and the confusion matrix on test.

    The matrix rows are the real values and the columns are the predicted values,
    in the order of model.classes_ (for 0 and 1: [[TN, FP], [FN, TP]]).
    """
    test_predictions = model.predict(X_test)
    return {
        "train_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
        "confusion_matrix": confusion_matrix(y_test, test_predictions, labels=model.classes_),
    }

def baseline_accuracy(y_train: pd.Series, y_test: pd.Series) -> float:
    """Accuracy of always predicting the most common value in train."""
    most_common = y_train.mode().iloc[0]
    return float((y_test == most_common).mean())