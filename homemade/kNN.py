import numpy as np

class KNeighborsClassifier:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        y_pred = []
        for x in X:
            # L2 distances
            dists = np.linalg.norm(self.X_train - x, axis=1)
            idx = np.argsort(dists)[:self.k]
            votes = self.y_train[idx]
            # vote
            counts = np.bincount(votes)
            y_pred.append(np.argmax(counts))
        return np.array(y_pred)
