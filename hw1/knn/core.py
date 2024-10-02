# 算法核心
from sklearn import datasets
from collections import Counter  # 为了做投票
from sklearn.model_selection import train_test_split
import numpy as np


class KNN:
    def __init__(self, X, Y, K):
        '''
        X: Traning Dataset
        Y: Corresponding Labels
        K: K-NN
        '''
        self.X = X
        self.Y = Y
        self.K = K

    def classify(x):
        '''
        x: an input data point
        return: the classifed result for the input x
        '''
        

    def regression():
        ...
