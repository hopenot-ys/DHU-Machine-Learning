from collections import Counter

import numpy as np


class KNN:
    def __init__(self, X, Y, K, P):
        self.X = X
        self.Y = Y
        self.K = K
        self.P = P

    @staticmethod
    def _minkowski_distance(x1, x2, p):
        return np.linalg.norm(x1 - x2, ord=p)

    def _get_k_nearest(self, x, dis_func):
        distances = [dis_func(i, x, self.P) for i in self.X]
        sorted_indices = np.argsort(distances)
        return sorted_indices[:self.K]

    def _classify(self, x, dis_func):
        k_labels = self.Y[self._get_k_nearest(x, dis_func)]
        return Counter(k_labels).most_common(1)[0][0]

    def _regress(self, x, dis_func):
        k_values = self.Y[self._get_k_nearest(x, dis_func)]
        return np.mean(k_values)

    def knn_classify(self, x):
        return self._classify(x, KNN._minkowski_distance)

    def knn_regress(self, x):
        return self._regress(x, KNN._minkowski_distance)
