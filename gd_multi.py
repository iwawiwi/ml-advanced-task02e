
import numpy as np
import random as rnd
import sys

def polynomial_2d(x1, x2):
    return x1 + x2


def createMultiX(x1, x2, m):
    X = np.zeros(shape=(len(x1), m+1))
    for k in range(m+1):
        # col = np.zeros(len(x1))
        col = []
        for a, b in zip(x1, x2):
            col.append(pow(polynomial_2d(a, b), k))
            # print 'col: ', col
        X[:, k] = col
    return np.mat(X)


def computeCost(X, y, w):
    n = y.size
    predictions = X * w
    err = predictions - y
    sqErr = np.power(err, 2)
    sqErrSum = sqErr.sum()
    J = (1.0 / (2*n)) * sqErrSum
    return J


def computeRMSE(err):
    sqErr = np.power(err, 2)
    sqErrSum = sqErr.sum()
    J = (1.0 / (2 * err.size)) * sqErrSum
    return J


def standMultiReg(x1, x2, y, m):
    x1 = np.mat(x1).flatten().T
    x2 = np.mat(x2).flatten().T
    X = createMultiX(x1, x2, m)
    y = np.mat(y).flatten().T
    XTX = X.T * X
    if np.linalg.det(XTX) == 0.0:
        print "The XTX matrix is singular, cannot do inverse"
        return
    w = XTX.I * X.T * y
    J = computeCost(X, y, w)
    return X, w, J


def gradDescentMulti(x1, x2, y, m, num_iters=1500, alpha=0.0001):
    x1 = np.mat(x1).flatten().T
    x2 = np.mat(x2).flatten().T
    X = createMultiX(x1, x2, m)
    y = np.mat(y).flatten().T
    n = y.size
    w = np.mat(np.zeros(shape=(m+1, 1)))
    E_history = np.zeros(shape=(num_iters, 1))
    prevE = float("inf")
    for i in range(num_iters):
        E = computeCost(X, y, w)
        if E < prevE:
            predictions = X * w
            diff = predictions - y
            for j in range(m+1):
                error = diff.T * X[:, j]
                w[j][0] += -alpha * (1.0/n) * error.sum()
            E_history[i, 0] = E  # computeCost(X, y, w)
            prevE = E
        else:
            print 'diverged! try using smaller alpha'
            sys.exit(1)
    return X, w, E_history


def stocGradDescentMulti(x1, x2, y, m, num_iters=1500, alpha=0.0001):
    x1 = np.mat(x1).flatten().T
    x2 = np.mat(x2).flatten().T
    X = createMultiX(x1, x2, m)
    y = np.mat(y).flatten().T
    n = y.size
    w = np.mat(np.zeros(shape=(m+1, 1)))
    E_history = np.zeros(shape=(num_iters, 1))
    for i in range(num_iters):
        dataIndex = rnd.randint(0, n-1)
        E = computeCost(X[dataIndex], y[dataIndex], w)
        predictions = X[dataIndex] * w
        diff = predictions - y[dataIndex]
        for j in range(m+1):
            error = diff.T * X[dataIndex, j]
            w[j][0] += -alpha * (1.0/n) * error
        E_history[i, 0] = E  # computeCost(X, y, w)
    return X, w, E_history


def stocGradDescentMulti2(x1, x2, y, m, num_iters=1500, alpha=0.0001, sample_size=4):
    x1 = np.mat(x1).flatten().T
    x2 = np.mat(x2).flatten().T
    X = createMultiX(x1, x2, m)
    y = np.mat(y).flatten().T
    n = y.size
    w = np.mat(np.zeros(shape=(m+1, 1)))
    E_history = np.zeros(shape=(num_iters, 1))
    for i in range(num_iters):
        randIndex = rnd.sample(range(len(X)), sample_size)
        E = computeCost(X[randIndex], y[randIndex], w)
        predictions = X[randIndex] * w
        diff = predictions - y[randIndex]
        for j in range(m+1):
            error = diff.T * X[randIndex, j]
            w[j][0] += -alpha * (1.0/n) * error.sum()
        E_history[i, 0] = E  # computeCost(X, y, w)
    return X, w, E_history


def conjugateGrad(x1, x2, y, m, num_iters=1500, threshold=0):
    """
    Logistic Regression using Conjugate Gradient

    Ax = b -> same as Xw = y in this context
    IMPORTANT: A must be a symetric and positive definite. If A is not symetric, A should be A.T * A.
    The original equation should be A.T * A * x = A.T * b
    """
    x1 = np.mat(x1).flatten().T
    x2 = np.mat(x2).flatten().T
    X = createMultiX(x1, x2, m)
    y = np.mat(y).flatten().T # prepare y.T
    w = np.mat(np.zeros(shape=(m+1, 1)))
    E_history = np.zeros(shape=(num_iters, 1))
    # algorithm start
    XTX = X.T * X # Positive definite matrix
    r = X.T * y - XTX * w # Equation to optimize
    p = r
    i = 0
    while i < num_iters:
        # for j in range(m + 1):
        # print p.T.shape, X.shape, p.shape, r.shape, r.T.shape
        xp = XTX * p
        rtr_old = r.T * r
        alpha = np.asscalar(rtr_old / (p.T * xp)) # alpha must be a scalar
        w += alpha * p
        r -= alpha * xp # this is the error
        RMSE = computeRMSE(r) # Compute RMSE
        # print RMSE
        E_history[i, 0] = RMSE # store error
        # print 'RMSE at iter-' + str(i) + ' = ' + str(RMSE)
        if RMSE <= threshold: # if error is tolerable
            break
        beta = np.asscalar(r.T * r / rtr_old) # beta must be a scalar
        p = r + (beta * p)
        i += 1
        # algorithm end
    return X, w, E_history

# def stocGradDescent2(x, y, m, num_iters=1500, alpha=0.0001, sample_size=4):
#     n = y.size
#     X = createX(x, m)  # create X
#     y = np.mat(y)
#     y = y.T  # prepare y
#     w = np.mat(np.zeros(shape=(m+1, 1)))
#     E_history = np.zeros(shape=(num_iters, 1))
#     for i in range(num_iters):
#         randIndex = rnd.sample(range(len(X)), sample_size)
#         E = computeCost(X[randIndex], y[randIndex], w)
#         predictions = X[randIndex] * w
#         diff = predictions - y[randIndex]
#         for j in range(m+1):
#             error = diff.T * X[randIndex, j]
#             w[j][0] = w[j][0] - alpha * (1.0/n) * error.sum()
#
#         E_history[i, 0] = E  # computeCost(X, y, w)
#     return w, E_history






