import os

import numpy as np


def whitenapply(X, m, P, dimensions=None):
    if not dimensions:
        dimensions = P.shape[1]

    X = np.dot(X - m, P[:, :dimensions])
    X = X / (np.linalg.norm(X, ord=2, axis=1, keepdims=True) + 1e-6)

    return X


def pcawhitenlearn(X):
    N = X.shape[0]

    # Learning PCA w/o annotations
    m = X.mean(axis=0, keepdims=True)
    Xc = X - m
    cov = np.dot(Xc.T, Xc) / N
    U, S, V = np.linalg.svd(cov)

    return m, U


def whitenlearn(X, qidxs, pidxs):
    # Learning Lw w annotations
    m = X[:, qidxs].mean(axis=1, keepdims=True)
    df = X[:, qidxs] - X[:, pidxs]
    S = np.dot(df, df.T) / df.shape[1]
    P = np.linalg.inv(cholesky(S))
    df = np.dot(P, X - m)
    D = np.dot(df, df.T)
    eigval, eigvec = np.linalg.eig(D)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    P = np.dot(eigvec.T, P)

    return m, P


def cholesky(S):
    # Cholesky decomposition
    # with adding a small value on the diagonal
    # until matrix is positive definite
    alpha = 0
    while 1:
        try:
            L = np.linalg.cholesky(S + alpha * np.eye(*S.shape))
            return L
        except:
            if alpha == 0:
                alpha = 1e-10
            else:
                alpha *= 10
            print(">>>> {}::cholesky: Matrix is not positive definite, adding {:.0e} on the diagonal"
                  .format(os.path.basename(__file__), alpha))
