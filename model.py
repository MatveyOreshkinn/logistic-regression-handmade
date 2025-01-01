import numpy as np
import pandas as pd
from typing import Optional
from scipy.special import expit


def sigmoid(x):
    return expit(x)


class MyLogReg:
    def __init__(self, n_iter: int = 10, learning_rate: float = 0.1, weights: Optional[np.ndarray] = None) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

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

            if verbose and i % verbose == 0:
                print(f'{i} | loss: {logloss:.2f}')

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_prob = self.predict_proba(X)
        return np.where(y_prob > 0.5, 1, 0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()  # чтобы не менять оригинальный dataframe
        X.insert(0, 'x0', 1)
        y_pred = np.dot(X, self.weights)
        return sigmoid(y_pred)

    def get_coef(self) -> np.ndarray:
        """
        Возвращает коэффициенты модели (без bias).

        Returns:
           np.ndarray: Вектор коэффициентов модели.

        """
        return self.weights[1:]
