import numpy as np
import pandas as pd
from typing import Optional, Any
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import random


def sigmoid(x):
    return expit(x)


class MyLogReg:
    def __init__(self,
                 n_iter: int = 10,
                 learning_rate: Any = 0.1,
                 weights: Optional[np.ndarray] = None,
                 metric: Optional[str] = None,
                 reg: Optional[str] = None,
                 l1_coef: float = 0,
                 l2_coef: float = 0,
                 sgd_sample: float = None,
                 random_state: int = 42) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.rangom_state = random_state
        self.sgd_sample = sgd_sample

    def __str__(self) -> str:
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Optional[bool] = False) -> None:
        random.seed(self.rangom_state)
        X.insert(0, 'x0', 1)  # оптимизация для нахождения градиента
        cnt_features = X.shape[1]

        if self.weights is None:
            self.weights = np.ones(cnt_features)

        for i in range(self.n_iter):
            if self.sgd_sample is None:
                y_pred = np.dot(X, self.weights)
                y_select = y
                X_select = X
            else:
                if isinstance(self.sgd_sample, float):
                    self.sgd_sample = round(X.shape[0] * self.sgd_sample)

                sample_rows_idx = random.sample(
                    range(X.shape[0]), self.sgd_sample)
                y_pred = np.dot(X, self.weights)[sample_rows_idx]
                y_select = y.iloc[sample_rows_idx]
                X_select = X.iloc[sample_rows_idx]

            y_prob = sigmoid(y_pred)

            # Добавляем небольшое значение для предотвращения log(0)
            epsilon = 1e-15
            logloss = -np.mean(y_select * np.log(y_prob + epsilon) + (1 - y_select)
                               * np.log(1 - y_prob + epsilon))

            regular = 0
            if self.reg is not None:
                if self.reg == 'l1':
                    regular = self.l1()
                elif self.reg == 'l2':
                    regular = self.l2()
                else:
                    regular = self.elasticnet()

            grad = (y_prob - y_select) @ X_select / X_select.shape[0] + regular

            if callable(self.learning_rate):
                self.weights -= grad * self.learning_rate(i + 1)
            else:
                self.weights -= grad * self.learning_rate

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

    def l1(self) -> np.ndarray:
        """
        Вычисляет L1-регуляризацию для текущих весов модели.

        Returns:
            np.ndarray: Значения L1-регуляризации, умноженные на коэффициент l1_coef.

        """
        return self.l1_coef * np.sign(self.weights)

    def l2(self) -> np.ndarray:
        """
        Вычисляет L2-регуляризацию для текущих весов модели.

        Returns:
            np.ndarray: Значения L2-регуляризации, умноженные на коэффициент l2_coef.

        """
        return self.l2_coef * 2 * self.weights

    def elasticnet(self) -> np.ndarray:
        """
        Вычисляет комбинацию L1 и L2-регуляризации (Elastic Net) для текущих весов модели.

        Returns:
            np.ndarray: Сумма L1 и L2-регуляризаций.

        """
        return self.l1() + self.l2()

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
