import numpy as np
from time import time

def true_fun(x1, x2):
    return np.sin(x1) + np.cos(x2)

# true signal
x1 = x2 = np.linspace(0.0, 2 * np.pi, 100, True)
x1, x2 = np.meshgrid(x1, x2)
y_true = true_fun(x1, x2)

# noisy signal
n_sample = 50
x1_sample = x2_sample = np.linspace(0.0, 2 * np.pi, n_sample, True)
x1_sample, x2_sample = np.meshgrid(x1_sample, x2_sample)
noise = [np.random.normal(0, 0.30, len(x)) for x in x1_sample]
signal = true_fun(x1_sample, x2_sample)
y_sample = signal + noise

import gd_multi as gdm

order = 9

# gradient descent methods
n_gdm_iter = 100000
gdm_alpha = 0.000000000000000001
sample_size = 10

t0 = time()

# ordinary GD
X, w_gdm, J_gdm_history = gdm.gradDescentMulti(x1_sample, x2_sample, y_sample, order,
                                               n_gdm_iter, gdm_alpha)
print 'w_gdm: ', w_gdm
y_gdm_approx = X * w_gdm
y_gdm_approx = np.reshape(y_gdm_approx, (len(x1_sample), -1))
t1 = time()

# conjugate gradient
X, w_cg, J_cg_history = gdm.conjugateGrad(x1_sample, x2_sample, y_sample, order, n_gdm_iter)
print 'w_cg: ', w_cg
y_cg_approx = X * w_cg
y_cg_approx = np.reshape(y_cg_approx, (len(x1_sample), -1))
t2 = time()

# Newton
X, w_newton, J_newton_history = gdm.newtonMulti(x1_sample, x2_sample, y_sample, order, n_gdm_iter)
print 'w_newton: ', w_newton
y_newton_approx = X * w_newton
y_newton_approx = np.reshape(y_newton_approx, (len(x1_sample), -1))
t2 = time()

ct_normal_eq = t1 - t0
ct_cg = t2 - t1


# plot
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# PLOT TITLE
title = '[n' + str(n_sample) + '][m' + str(order) + '][iter' + str(n_gdm_iter) + ']'

# plot results
fig = plt.figure(title + 'Result Objective')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1, x2, y_true, cmap=cm.hot, alpha=0.3)
ax.scatter(x1_sample, x2_sample, y_sample, s=1, c='b')
# ax.scatter(x1_sample, x2_sample, y_sample, s=1)
ax.set_zlim(-2.5, 2.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('E(x1, x2)=sin(x1) + cos(x2)')
plt.title('The True Signal and Data Samples')
# plt.show()

# plot normal NEWTON  result
fig = plt.figure(title + 'Result NEWTON')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_sample, x2_sample, y_newton_approx, cmap = cm.hot, alpha=0.3)
ax.set_zlim(-2.5, 2.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.title('Polynomial Approx. using Newton Method')
# plt.show()

# plot error profile
n_iter = len(J_gdm_history)
x_error = np.arange(n_iter)
fig = plt.figure(title + 'Error Profile')
ax = fig.add_subplot(111)
ax.plot(x_error, J_gdm_history, 'g--', linewidth=2, label='GDM')
ax.plot(x_error, J_cg_history, 'y--', linewidth=2, label='CG')
ax.plot(x_error, J_newton_history, 'r--', linewidth=2, label='Newton')
plt.ylabel('Error Profile')
plt.xlabel('Iteration')
plt.title('GDM,CG Error Profile with M = ' + str(order) + ', N = ' + str(len(y_gdm_approx)))
plt.legend()
plt.grid()
# plt.show()

plt.show()
