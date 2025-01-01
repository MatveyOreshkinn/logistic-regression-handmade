import numpy as np
import pandas as pd
from typing import Optional
from scipy.special import expit
from sklearn.metrics import roc_auc_score


def sigmoid(x):
    return expit(x)


class MyLogReg:
    def __init__(self, n_iter: int = 10, learning_rate: float = 0.1,
                 weights: Optional[np.ndarray] = None, metric: Optional[str] = None) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = None

    def __str__(self) -> str:
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Optional[bool] = False) -> None:
        X = X.copy()  # чтобы не менять оригинальный dataframe
        X.insert(0, 'x0', 1)  # оптимизация для нахождения градиента
        cnt_features = X.shape[1]

        if self.weights is None:
            self.weights = np.ones(cnt_features)

        for i in range(self.n_iter):
            y_pred = np.dot(X, self.weights)
            y_prob = sigmoid(y_pred)

            # Добавляем небольшое значение для предотвращения log(0)
            epsilon = 1e-15
            y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
            logloss = -np.mean(y * np.log(y_prob) + (1 - y)
                               * np.log(1 - y_prob))

            grad = (y_prob - y) @ X / X.shape[0]
            self.weights -= self.learning_rate * grad

            if self.metric is not None:
                if self.metric == 'accuracy':
                    metric_value = self.accuracy(X.drop('x0', axis=1), y)
                elif self.metric == 'precision':
                    metric_value = self.precision(X.drop('x0', axis=1), y)
                elif self.metric == 'recall':
                    metric_value = self.recall(X.drop('x0', axis=1), y)
                elif self.metric == 'f1':
                    metric_value = self.f1(X.drop('x0', axis=1), y)
                elif self.metric == 'roc_auc':
                    metric_value = self.roc_auc(X.drop('x0', axis=1), y)

                if self.best_score is None or metric_value > self.best_score:
                    self.best_score = metric_value

            if verbose and i % verbose == 0:
                print(f'{i} | loss: {logloss:.2f}| {self.metric}: {metric_value}')

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_prob = self.predict_proba(X)
        return np.where(y_prob > 0.5, 1, 0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()  # чтобы не менять оригинальный dataframe
        X.insert(0, 'x0', 1)
        y_pred = np.dot(X, self.weights)
        return sigmoid(y_pred)

    def accuracy(self, X: pd.DataFrame, y: pd.Series) -> float:
        pred = self.predict(X)
        return np.mean(pred == y)

    def precision(self, X: pd.DataFrame, y: pd.Series) -> float:
        pred = self.predict(X)
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        if tp + fp == 0:
            return 0  # Избегаем деления на ноль
        return tp / (tp + fp)

    def recall(self, X: pd.DataFrame, y: pd.Series) -> float:
        pred = self.predict(X)
        tp = np.sum((pred == 1) & (y == 1))
        fn = np.sum((pred == 0) & (y == 1))
        if tp + fn == 0:
            return 0  # Избегаем деления на ноль
        return tp / (tp + fn)

    def f1(self, X: pd.DataFrame, y: pd.Series) -> float:
        pr = self.precision(X, y)
        re = self.recall(X, y)
        if pr + re == 0:
            return 0  # Избегаем деления на ноль
        return (2 * pr * re) / (pr + re)

    def roc_auc(self, X: pd.DataFrame, y: pd.Series) -> float:
        y_prob = self.predict_proba(X)
        return roc_auc_score(y, y_prob)

    def get_best_score(self):
        return self.best_score

    def get_coef(self) -> np.ndarray:
        """
        Возвращает коэффициенты модели (без bias).

        Returns:
           np.ndarray: Вектор коэффициентов модели.

        """
        return self.weights[1:]
