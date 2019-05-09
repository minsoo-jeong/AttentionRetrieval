import os
import torch
import numpy as np


def whitenapply(X, m, P, dimensions=None):
    if not dimensions:
        dimensions = P.shape[1]

    X = np.dot(X - m, P[:, :dimensions])
    X = X / (np.linalg.norm(X, ord=2, axis=1, keepdims=True) + 1e-12)

    return X


def pcawhitenlearn(X):
    N = X.shape[0]

    # Learning PCA w/o annotations
    m = X.mean(axis=0, keepdims=True)
    Xc = X - m
    Xcov = np.dot(Xc.T, Xc)
    Xcov = (Xcov + Xcov.T) / (2 * N)
    eigval, eigvec = np.linalg.eig(Xcov)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    P = np.dot(np.linalg.inv(np.sqrt(np.diag(eigval))), eigvec.T)

    return m, P


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


def learningPCA2(listData):
    fudge = 1E-18
    X=listData
    mean = X.mean(axis=0)
    # subtract the mean
    X = np.subtract(X, mean)
    # calc covariance matrix
    Xcov = np.dot(X.T,X)

    U,S,V=np.linalg.svd(Xcov)

    return U,S,V,mean

def apply_whitening2(X,m,u,s):
    X=torch.Tensor(X)
    m = torch.Tensor(m)
    u = torch.Tensor(u)
    s = torch.Tensor(s)

    x_whiten=torch.mm(torch.sub(X,m),u[:,:])/torch.sqrt(s[:]+1e-18)
    x_whiten /= torch.norm(x_whiten,p=2,dim=1,keepdim=True)
    return x_whiten.numpy()