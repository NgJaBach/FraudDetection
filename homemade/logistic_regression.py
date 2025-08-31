import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.1, n_iters=5000):
        self.lr = lr
        self.n_iters = n_iters
        self.w_ = None
        self.b_ = 0.0

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _add_intercept(self, X):
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([ones, X])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        uniq = np.unique(y)
        if set(uniq.tolist()) == {-1.0, 1.0}:
            y = (y + 1.0) / 2.0

        X_aug = self._add_intercept(X)
        n_features = X_aug.shape[1]

        rng = np.random.default_rng(42)
        theta = rng.normal(scale=0.01, size=n_features)

        m = X_aug.shape[0]

        for i in range(self.n_iters):
            z = X_aug @ theta
            y_hat = self._sigmoid(z)

            # Gradient of average log-loss
            error = (y_hat - y)  # shape (m,)
            grad = (X_aug.T @ error) / m  # shape (n_features,)

            theta -= self.lr * grad

        self.b_ = float(theta[0])
        self.w_ = theta[1:].copy()

        return self
    
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        z = X @ self.w_ + self.b_
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)