from __future__ import print_function, division
import numpy as np


class KNN():

    def __init__(self, k=5):
        self.k = k

    def predict(self, X_test, X_train, y_train):
        """
        X_test: [N, C]
        X_train: [M, C]
        y_train: [M]
        """
        N, M = X_test.shape[0], X_train.shape[0]
        X_test = np.expand_dims(X_test, axis=1)  # [N, 1, C]
        X_test = np.repeat(X_test, M, axis=1)  # [N, M, C]
        X_train = np.expand_dims(X_train, axis=0)  # [1, M, C]
        X_train = np.repeat(X_train, N, axis=0)  # [N, M, C]

        distances = np.sum((X_train - X_test) ** 2, axis=2)  # [N, M]
        sorted_indices = np.argsort(distances, axis=1)[:, :self.k]  # [N, k]

        y_train = y_train.astype(np.int32)
        y_pred = np.zeros((N,), dtype=np.int32)
        for i in range(N):
            neighbors = y_train[sorted_indices[i]]  # [k]
            bin = np.bincount(neighbors)
            y_pred[i] = np.argmax(bin, axis=0)

        return y_pred


if __name__ == '__main__':
    model = KNN()
    X_test = np.random.rand(128, 8)
    X_train = np.random.rand(32, 8)
    y_train = np.random.randint(0, 3, size=(128,))
    y_pred = model.predict(X_test, X_train, y_train)
    
    print(y_pred)
    print(y_pred.shape)
