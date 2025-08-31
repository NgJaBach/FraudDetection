import numpy as np

class GaussianNB:
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)

        self.class_count_ = np.bincount(y_idx, minlength=n_classes)
        self.class_prior_ = self.class_count_ / y.size

        self.theta_ = np.vstack([X[y_idx == k].mean(axis=0) for k in range(n_classes)])
        var = np.vstack([X[y_idx == k].var(axis=0) for k in range(n_classes)])
        eps = 1e-9 * X.var(axis=0).max() + 1e-12
        self.var_ = var + eps
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # Joint log-likelihood
        const = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_), axis=1) # (C,)
        quad  = -0.5 * ((X[:, None, :] - self.theta_)**2 / self.var_).sum(2) # (N,C)
        jll = quad + const + np.log(self.class_prior_) # (N,C)
        return self.classes_[np.argmax(jll, axis=1)]
