import numpy as np


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x_test):
        y_pred = [self.get_single_prediction(x_test_row) for x_test_row in x_test]
        return np.array(y_pred)

    def get_single_prediction(self, x_test_row):
        distances = [self._get_euclidean_distance(x_test_row, x_train_row) for x_train_row in self.x_train]
        # get indices of k-nearest neighbors -> k-smallest distances
        k_idx = np.argsort(distances)[:self.k]
        # get corresponding y-labels of training data
        k_labels = [self.y_train[idx] for idx in k_idx]
        # return most-common label
        return np.argmax(np.bincount(k_labels))

    def _get_euclidean_distance(self, x1, x2):
        sum_squared_distance = np.sum((x1 - x2) ** 2)
        return np.sqrt(sum_squared_distance)
