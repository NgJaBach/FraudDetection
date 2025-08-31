import numpy as np

def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = y_true.size
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true != 1) & (y_pred != 1))
    fp = np.sum((y_true != 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred != 1))

    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    print(f"Accuracy             : {acc:.6f}")
    print(f"Precision (pos=1)    : {prec:.6f}")
    print(f"Recall/TPR (pos=1)   : {rec:.6f}")
    print(f"F1 (pos=1)           : {f1:.6f}")

def train_test_split(X, y, stratify, test_size, random_state, shuffle=True):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X); y = np.asarray(y)
    n = y.shape[0]

    s = np.asarray(stratify)
    r = (test_size if isinstance(test_size, float) else test_size / n)
    test_idx, train_idx = [], []
    for cls in np.unique(s):
        cls_idx = np.where(s == cls)[0]
        if shuffle: rng.shuffle(cls_idx)
        m_cls = int(np.ceil(r * len(cls_idx)))
        m_cls = max(0, min(len(cls_idx) - 1, m_cls))  # keep at least 1 for train if possible
        test_idx.extend(cls_idx[:m_cls])
        train_idx.extend(cls_idx[m_cls:])
    test_idx, train_idx = np.array(test_idx), np.array(train_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
