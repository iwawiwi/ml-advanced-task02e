
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


def computeRMSE(err):
    sqErr = np.power(err, 2)
    sqErrSum = sqErr.sum()
    J = (1.0 / (2 * err.size)) * sqErrSum
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
                w[j][0] -= alpha * (1.0 / n) * error.sum()
            E_history[i, 0] = E  # computeCost(X, y, w)
            prevE = E
        else:
            print 'diverged! try using smaller alpha'
    return w, E_history


def gradDescentNewton(x, y, m, num_iters=1500):
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
                XTX = X[:,j].T * X[:,j]
                w[j][0] -= (1.0 / n) * error.sum() * XTX.I
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
            w[j][0] -= alpha * (1.0 / n) * error

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
            w[j][0] -= alpha * (1.0 / n) * error.sum()

        E_history[i, 0] = E  # computeCost(X, y, w)
    return w, E_history


# def gdSimAnneal(x, y, m, num_iters=1500, alpha=0.0001):
#     """
#     GD with Simulated Annealing optimization
#     """
#     n = y.size
#     X = createX(x, m)
#     y = np.mat(y)
#     y = y.T # prepare y
#     w = np.mat(np.zeros(shape=(m+1, 1)))
#     E_history = np.zeros(shape=(num_iters, 1))
#     prevE = float("inf")
#
#     # TODO: This is the iteration to optimize
#     for i in range(num_iters):
#         E = computeCost(X, y, w)
#         if E < prevE:
#             predictions = X * w
#             diff = predictions - y
#             for j in range(m+1):
#                 error = diff.T * X[:, j]
#                 w[j][0] -= alpha * (1.0 / n) * error.sum()
#             E_history[i, 0] = E  # computeCost(X, y, w)
#             prevE = E
#         else:
#             print 'diverged! try using smaller alpha'
#     return w, E_history


# def conjugateGrad(x, y, m, num_iters=1500, threshold=0.0000000000001):
#     """
#     Logistic Regression using Conjugate Gradient
#     Taken from http://www.omidrouhani.com/research/logisticregression/html/logisticregression.htm#_Toc147483473
#
#     Ax = b -> same as Xw = y in this context
#     """
#     X = createX(x, m)
#     y = np.mat(y)
#     y = y.T # prepare y.T
#     w = np.mat(np.zeros(shape=(m+1, 1)))
#     E_history = np.zeros(shape=(num_iters, 1))
#     # algorithm start
#     i = 0
#     r = y - X * w
#     d = r
#     delta_current = np.asscalar(r.T * r)
#     # print str(delta_current) + ' > ' + str(threshold) + ' and ' + str(i) + ' < ' + str(num_iters)
#     # delta0 = delta_current # TODO: Unused?
#     while delta_current > threshold and i < num_iters:
#         print np.size(X.T,0), np.size(X,1), np.size(d,0), np.size(d,1)
#         q = X.T * d
#         print np.size(d,0), np.size(d,1), np.size(q,0), np.size(q,1)
#         alpha = np.asscalar(delta_current / (d.T * q))
#         w += alpha * d
#         r -= alpha * q# this is current error
#         E_history[i, 0] = r # store error history
#         delta_old = delta_current
#         delta_current = np.asscalar(r.T * r)
#         beta = delta_current / delta_old
#         d = r + beta * d
#         i += 1
#     # algorithm end
#     return w, E_history

def conjugateGrad(x, y, m, num_iters=1500, threshold=0):
    """
    Logistic Regression using Conjugate Gradient

    Ax = b -> same as Xw = y in this context
    IMPORTANT: A must be a symetric and positive definite. If A is not symetric, A should be A.T * A.
    The original equation should be A.T * A * x = A.T * b
    """
    X = createX(x, m)
    y = np.mat(y)
    y = y.T # prepare y.T
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
        if RMSE <= threshold: # if error is tolerable
            break # quit iteration
        beta = np.asscalar(r.T * r / rtr_old) # beta must be a scalar
        p = r + (beta * p)
        i += 1
    # algorithm end
    return w, E_history


def simulatedAnneal(x, y, m, num_iters=1500):
    n = y.size
    X = createX(x, m)  # create X
    y = np.mat(y)
    y = y.T  # prepare y
    w = np.mat(np.zeros(shape=(m+1, 1)))

    w, E_history = annealingSchedule(X, y, w, num_iters=num_iters) # compute error based on annealing schedule

    # update w
    # predictions = X * w # best prediction
    # diff = predictions - y
    # for j in range(np.size(w,0)):
    #     error = diff.T * X[:, j]
    #     print 'error', error.sum()
    #     w[j][0] -= (1.0 / n) * error.sum()

    # sbest, E_history = annealingSchedule(X, y, w, num_iters=num_iters) # compute error based on annealing schedule
    # predictions = X[sbest] * w # best prediction
    # diff = predictions - y
    # for j in range(m+1):
    #     error = diff.T * X[:, j]
    #     w[j][0] -= (1.0 / n) * error.sum()

    print 'w: ', w, 'E:', E_history
    return w, E_history


def annealingSchedule(X, y, w, num_iters=1500, threshold=0, anneal_param=0.95, max_temp=150):
    n = y.size
    # X = createX(x, m)
    # y = np.mat(y)
    # y = y.T # prepare y.T
    # w = np.mat(np.zeros(shape=(m+1, 1)))
    E_history = np.zeros(shape=(num_iters, 1))
    # Algorithm start
    s = rnd.randint(0, n-1) # randomize initial state
    e = computeCost(X[s], y[s], w) # Energy of state
    sbest = s
    ebest = e
    i = 0
    while i < num_iters and e > threshold and e != float('inf'):
        temp = max_temp * pow(anneal_param, i) # calculate temprature -- default MATLAB
        snew = rnd.randint(0, n-1) # select neighborhood randomly
        enew = computeCost(X[snew], y[snew], w)

        # print 'enew = ' + str(enew) + ', e = ' + str(e) + ', temp = ' + str(temp)
        # Acceptace of move function
        if enew == float('inf'):
            break # break loop if energy too high or inifinite

        accept_prob = np.float(1/(1 + (np.exp(enew - e)/temp))) # default MATLAB

        if np.random.random_sample() < accept_prob:
            s = snew
            e = enew

            # update w
            predictions = X * w # best prediction
            diff = predictions - y
            print 'diff', diff
            for j in range(np.size(w,0)):
                error = diff.T * X[:, j]
                print 'error', error.sum()
                w[j][0] -= (1.0 / n) * error.sum()

        if enew < ebest:
            sbest = snew
            ebest = enew
        E_history[i, 0] = e
        i += 1
    # Algorithm end
    return w, E_history # return minimum error found


def createModel(x, w):
    y = np.zeros(len(x))
    y = np.mat(y)
    y = y.T  # prepare y
    pwr = np.arange(len(w))
    for wi, p in zip(w, pwr):
        accum = np.mat(wi * (x ** p))
        y = y + accum.T
    return np.squeeze(np.asarray(y))




