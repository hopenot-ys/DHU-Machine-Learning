from collections import Counter
import numpy as np


class KNN:
    def __init__(self, X, Y, K, P, NORM):
        '''
        X: Traning Dataset
        Y: Corresponding Labels
        K: K-NN
        '''
        self.X = X
        self.Y = Y
        self.K = K
        self.P = P

        if NORM == 'none':
            self.NORM_FUNC = KNN._no_scale
        elif NORM == 'min_max_scale':
            self.NORM_FUNC = KNN._min_max_scale
        elif NORM == 'standard_scale':
            self.NORM_FUNC = KNN._standard_scale
        else:
            raise Exception("The wrong norm")

        self.X = np.array([self.NORM_FUNC(x) for x in self.X])

    @staticmethod
    def _minkowski_distance(x1, x2, p):
        return np.linalg.norm(x1 - x2, ord=p)

    def _get_k_nearest(self, x, dis_func):
        distances = [dis_func(i, x, self.P) for i in self.X]
        sorted_indices = np.argsort(distances)
        return sorted_indices[:self.K]

    def _classify(self, x, dis_func):
        '''
        x: an input data point
        return: the classifed result for the input x
        '''
        k_lables = self.Y[self._get_k_nearest(x, dis_func)]
        return Counter(k_lables).most_common(1)[0][0]

    def _regress(self, x, dis_func):
        k_values = self.Y[self._get_k_nearest(x, dis_func)]
        return np.mean(k_values)

    def knn_classify(self, x):
        return self._classify(self.NORM_FUNC(x), KNN._minkowski_distance)

    def knn_regress(self, x):
        return self._regress(self.NORM_FUNC(x), KNN._minkowski_distance)

    @staticmethod
    def _no_scale(x):
        return x

    @staticmethod
    def _min_max_scale(x):
        return np.array((x - min(x)) / (max(x) - min(x)))

    @staticmethod
    def _standard_scale(x):
        return np.array((x - np.mean(x)) / np.std(x))
