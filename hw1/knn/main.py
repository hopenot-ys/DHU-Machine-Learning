import numpy as np

from sklearn import datasets
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from core import KNN


def iris_dataset_load():
    iris: Bunch = datasets.load_iris()
    X = iris.data
    Y = iris.target
    return X, Y


def diabetes_dataset_load():
    diabetes: Bunch = datasets.load_diabetes()
    X = diabetes.data
    Y = diabetes.target
    return X, Y


def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def main():
    print("*" * 4 + " Classification " + "*" * 4)
    X, Y = iris_dataset_load()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=2003)
    for p in [1, 2, np.inf]:
        for norm in ['none', 'min_max_scale', 'standard_scale']:
            knn = KNN(X_train, y_train, 3, p, norm)
            y_predict = [knn.knn_classify(x) for x in X_test]
            accuracy = accuracy_score(y_test, y_predict)
            print(f"P: {p}, Norm: {norm} => Acc: {accuracy}")

    print("*" * 4 + " Regression " + "*" * 4)
    X, Y = diabetes_dataset_load()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=2003)
    for p in [1, 2, np.inf]:
        for norm in ['none', 'min_max_scale', 'standard_scale']:
            knn = KNN(X_train, y_train, 3, p, norm)
            y_predict = [knn.knn_regress(x) for x in X_test]
            mse = rmse(y_test, y_predict)
            print(f"P: {p}, Norm: {norm} => RMSE: {mse}")


if __name__ == '__main__':
    main()
