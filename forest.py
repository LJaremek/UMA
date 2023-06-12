from tree import DecisionTreeClassifier

import numpy as np


class RandomForestClassifier:
    """
    Implementacja podstawowego lasu losowego służącego do klasyfikacji.
    Na las składają się zaimplementowane przez nas drzewa

    Hiperparametry:
        - max_depth (int): maksymalna głębokość drzewa
        - min_samples_split (int): liczba próbek konieczna do podziału drzewa
        - n_estimators (int): liczba drzew

    Na wejściu wymagane wartości liczbowe.

    Przykład użycia:

    >>> from forest import RandomForestClassifier
    >>> ...
    >>> X_train, y_train = np.array([[2, 2], [0, 6]]), np.array([0, 1])
    >>> X_test = np.array([[2, 1])
    >>> ...
    >>> model = RandomForestClassifier()
    >>> model.fit(X_train, y_train)
    >>> model.predict(X_test)
    ...
    ...
    array([0.])

    """
    def __init__(self, n_estimators: int = 10, max_depth: int = None,
                 min_samples_split: int = 2) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X: np.array, y: np.array) -> None:
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split)

            bootstrap_indices = np.random.choice(
                len(X), size=len(X), replace=True)
            bootstrap_X = X[bootstrap_indices]
            bootstrap_y = y[bootstrap_indices]

            tree.fit(bootstrap_X, bootstrap_y)
            self.trees.append(tree)

    def predict(self, X: np.array) -> np.array:
        predictions = np.zeros(len(X), dtype=int)

        for tree in self.trees:
            tree_predictions = tree.predict(X)
            predictions += tree_predictions

        predictions = np.round(predictions / self.n_estimators)
        return predictions
