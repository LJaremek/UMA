import numpy as np


class DecisionTreeClassifier:
    """
    Implementacja prostego drzewa decyzyjnego służącego do klasyfikacji.
    Do podziału na poddrzewa, wykorzystywany jest information gain,
    bazujący na entropii.

    Hiperparametry:
        - max_depth (int): maksymalna głębokość drzewa
        - min_samples_split (int): liczba próbek konieczna do podziału

    Na wejściu wymagane wartości liczbowe.

    Przykład użycia:

    >>> from tree import DecisionTreeClassifier
    >>> ...
    >>> X_train, y_train = np.array([[2, 2], [0, 6]]), np.array([0, 1])
    >>> X_test = np.array([[2, 1])
    >>> ...
    >>> model = DecisionTreeClassifier()
    >>> model.fit(X_train, y_train)
    >>> model.predict(X_test)
    ...
    ...
    array([0.])
    """
    def __init__(self, max_depth: int = None,
                 min_samples_split: int = 2) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def entropy(self, y: np.array) -> float:
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def information_gain(self, X: np.array, y: np.array,
                         feature_index: int, threshold: float) -> float:
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        left_entropy = self.entropy(y[left_mask])
        right_entropy = self.entropy(y[right_mask])

        left_weight = len(y[left_mask]) / len(y)
        right_weight = len(y[right_mask]) / len(y)

        information_gain = self.entropy(y) -\
            (left_weight * left_entropy + right_weight * right_entropy)

        return information_gain

    def find_best_split(self, X: np.array, y: np.array) -> tuple:
        best_gain = 0
        best_feature_index = None
        best_threshold = None

        n_features = X.shape[1]

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gain = self.information_gain(X, y, feature_index, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def build_tree(self, X: np.array, y: np.array, depth: int = 0) -> dict:
        if len(np.unique(y)) == 1:
            return {'class': y[0]}

        if self.max_depth is not None and depth >= self.max_depth:
            return {'class': np.bincount(y).argmax()}

        if len(X) < self.min_samples_split:
            return {'class': np.bincount(y).argmax()}

        best_feature_index, best_threshold = self.find_best_split(X, y)

        if best_feature_index is None or best_threshold is None:
            return {'class': np.bincount(y).argmax()}

        left_mask = X[:, best_feature_index] <= best_threshold
        right_mask = ~left_mask

        left_tree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'feature_index': best_feature_index,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }

    def fit(self, X: np.array, y: np.array) -> None:
        self.tree = self.build_tree(X, y)

    def predict(self, X: np.array) -> np.array:
        predictions = []
        for sample in X:
            node = self.tree
            while 'class' not in node:
                if sample[node['feature_index']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            predictions.append(node['class'])
        return np.array(predictions)
