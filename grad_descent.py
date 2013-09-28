
import numpy as np
import random as rnd


def createX(x, m):
    X = np.zeros(shape=((len(x), m + 1)))
    row_index = 0
    for x_elem in x:
        xrow = []
        for i in range(m + 1):
            xrow.append(pow(x_elem, i))
        X[row_index:] = xrow
        row_index += 1
    # print 'rank: ', np.rank(np.mat(X))
    return np.mat(X)  # return matrix instead of ndarray


def computeCost(X, y, w):
    n = y.size
    predictions = X * w
    err = predictions - y
    sqErr = np.power(err, 2)
    sqErrSum = sqErr.sum()
    J = (1.0 / (2*n)) * sqErrSum
    return J


def standReg(x, y, m):
    X = createX(x, m)  # create X
    y = np.mat(y)
    y = y.T  # prepare y
    XTX = X.T * X
    if np.linalg.det(XTX) == 0.0:
        print "The XTX matrix is singular, cannot do inverse"
        return
    w = XTX.I * X.T * y
    print 'w_stand: ', w
    J = computeCost(X, y, w)
    return w, J


def gradDescent(x, y, m, num_iters=1500, alpha=0.0001):
    n = y.size
    X = createX(x, m)  # create X
    y = np.mat(y)
    y = y.T  # prepare y
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
                w[j][0] = w[j][0] - alpha * (1.0/n) * error.sum()
            E_history[i, 0] = E  # computeCost(X, y, w)
            prevE = E
        else:
            print 'diverged! try using smaller alpha'
    return w, E_history

def stocGradDescent(x, y, m, num_iters=1500, alpha=0.0001):
    n = y.size
    X = createX(x, m)  # create X
    y = np.mat(y)
    y = y.T  # prepare y
    w = np.mat(np.zeros(shape=(m+1, 1)))
    E_history = np.zeros(shape=(num_iters, 1))
    for i in range(num_iters):
        dataIndex = rnd.randint(0, n-1)
        E = computeCost(X[dataIndex], y[dataIndex], w)
        predictions = X[dataIndex] * w
        diff = predictions - y[dataIndex]
        for j in range(m+1):
            error = diff.T * X[dataIndex, j]
            w[j][0] = w[j][0] - alpha * (1.0/n) * error

        E_history[i, 0] = E  # computeCost(X, y, w)
    return w, E_history


def stocGradDescent2(x, y, m, num_iters=1500, alpha=0.0001, sample_size=4):
    n = y.size
    X = createX(x, m)  # create X
    y = np.mat(y)
    y = y.T  # prepare y
    w = np.mat(np.zeros(shape=(m+1, 1)))
    E_history = np.zeros(shape=(num_iters, 1))
    for i in range(num_iters):
        randIndex = rnd.sample(range(len(X)), sample_size)
        E = computeCost(X[randIndex], y[randIndex], w)
        predictions = X[randIndex] * w
        diff = predictions - y[randIndex]
        for j in range(m+1):
            error = diff.T * X[randIndex, j]
            w[j][0] = w[j][0] - alpha * (1.0/n) * error.sum()

        E_history[i, 0] = E  # computeCost(X, y, w)
    return w, E_history


def createModel(x, w):
    y = np.zeros(len(x))
    y = np.mat(y)
    y = y.T  # prepare y
    pwr = np.arange(len(w))
    for wi, p in zip(w, pwr):
        accum = np.mat(wi * (x ** p))
        y = y + accum.T
    return np.squeeze(np.asarray(y))




