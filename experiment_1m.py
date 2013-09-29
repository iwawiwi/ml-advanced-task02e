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

# normal equation method
t0 = time()
X, w_stand, J_stand = gdm.standMultiReg(x1_sample, x2_sample, y_sample, order)
print 'w_stand, J_stand: ', w_stand, ', ', J_stand
y_stand_approx = X * w_stand
y_stand_approx = np.reshape(y_stand_approx, (len(x1_sample), -1))
t1 = time()

# gradient descent methods
n_gdm_iter = 10
gdm_alpha = 0.000000000000000001
sample_size = 10

# ordinary GD
X, w_gdm, J_gdm_history = gdm.gradDescentMulti(x1_sample, x2_sample, y_sample, order,
                                               n_gdm_iter, gdm_alpha)
print 'w_gdm: ', w_gdm
y_gdm_approx = X * w_gdm
y_gdm_approx = np.reshape(y_gdm_approx, (len(x1_sample), -1))
t2 = time()

# stochastic GD
X, w_sgdm, J_sgdm_history = gdm.stocGradDescentMulti(x1_sample, x2_sample, y_sample, order,
                                                     n_gdm_iter, gdm_alpha)
print 'w_sgdm: ', w_sgdm
y_sgdm_approx = X * w_sgdm
y_sgdm_approx = np.reshape(y_sgdm_approx, (len(x1_sample), -1))
t3 = time()

# stochastic GD_2
X, w_sgdm2, J_sgdm2_history = gdm.stocGradDescentMulti2(x1_sample, x2_sample, y_sample, order,
                                                        n_gdm_iter, gdm_alpha, sample_size)
print 'w_sgdm2: ', w_sgdm2
y_sgdm2_approx = X * w_sgdm2
y_sgdm2_approx = np.reshape(y_sgdm2_approx, (len(x1_sample), -1))
t4 = time()

# conjugate gradient
X, w_cg, J_cg_history = gdm.conjugateGrad(x1_sample, x2_sample, y_sample, order, n_gdm_iter)
print 'w_cg: ', w_cg
y_cg_approx = X * w_cg
y_cg_approx = np.reshape(y_cg_approx, (len(x1_sample), -1))
t5 = time()

ct_normal_eq = t1 - t0
ct_grad_desc = t2 - t1
ct_stoc_grad_desc = t3 - t2
ct_stoc_grad_desc2 = t4 - t3
ct_cg = t5 - t4


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

# plot normal equation result
fig = plt.figure(title + 'Result NE')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_sample, x2_sample, y_stand_approx, cmap = cm.hot, alpha=0.3)
ax.set_zlim(-2.5, 2.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.title('Polynomial Approx. using Normal Equations')
# plt.show()

# plot GD results
fig = plt.figure(title + 'Result GD')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_sample, x2_sample, y_gdm_approx, cmap=cm.hot, alpha=0.3)
ax.set_zlim(-2.5, 2.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.title('GD Polynomial Approx.: '+str(n_gdm_iter)+' iterations, alpha = '+str(gdm_alpha))
# plt.show()

# plot SGD results
fig = plt.figure(title + 'Result SGD')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_sample, x2_sample, y_sgdm_approx, cmap=cm.hot, alpha=0.3)
ax.set_zlim(-2.5, 2.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.title('SGD Polynomial Approx.: '+str(n_gdm_iter)+' iterations, alpha = '+str(gdm_alpha))
# plt.show()

# plot SGD2 results
fig = plt.figure(title + 'Result SGD2')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_sample, x2_sample, y_sgdm2_approx, cmap=cm.hot, alpha=0.3)
ax.set_zlim(-2.5, 2.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.title('SGD2 Polynomial Approx.: '+str(n_gdm_iter)+' iterations, alpha = '+str(gdm_alpha))
# plt.show()

# plot CG result
fig = plt.figure(title + 'Result CG')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_sample, x2_sample, y_cg_approx, cmap=cm.hot, alpha=0.3)
ax.set_zlim(-2.5, 2.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.title('CG Polynomial Approx.: '+str(n_gdm_iter)+' iterations')

# plot GD error profile
n_iter = len(J_gdm_history)
x_error = np.arange(n_iter)
J_error = [J_stand] * n_iter
fig = plt.figure(title + 'Error Profile GD')
ax = fig.add_subplot(111)
ax.plot(x_error, J_error, 'r--', linewidth=2, label='Stand')
ax.plot(x_error, J_gdm_history, 'g--', linewidth=2, label='GDM')
ax.plot(x_error, J_cg_history, 'y--', linewidth=2, label='CG')
plt.ylabel('Error Profile')
plt.xlabel('Iteration')
plt.title('GDM,CG Error Profile with M = ' + str(order) + ', N = ' + str(len(y_gdm_approx)))
plt.legend()
plt.grid()
# plt.show()

# plot SGD error profile
n_iter = len(J_sgdm_history)
x_error = np.arange(n_iter)
J_error = [J_stand] * n_iter
fig = plt.figure(title + 'SGD Error Profile')
ax = fig.add_subplot(111)
ax.plot(x_error, J_error, 'r--', linewidth=2, label='Stand')
ax.plot(x_error, J_sgdm_history, 'g--', linewidth=2, label='SGDM')
plt.ylabel('Error Profile')
plt.xlabel('Iteration')
plt.title('SGDM Error Profile with M = ' + str(order) + ', N = ' + str(len(y_sgdm_approx)))
plt.legend()
plt.grid()
# plt.show()

# plot SGD2 error profile
n_iter = len(J_sgdm2_history)
x_error = np.arange(n_iter)
J_error = [J_stand] * n_iter
fig = plt.figure(title + 'SGD2 Error Profile')
ax = fig.add_subplot(111)
ax.plot(x_error, J_error, 'r--', linewidth=2, label='Stand')
ax.plot(x_error, J_sgdm2_history, 'g--', linewidth=2, label='SGDM')
plt.ylabel('Error Profile')
plt.xlabel('Iteration')
plt.title('SGDM2 Error Profile with M = ' + str(order) + ', N = ' + str(len(y_sgdm2_approx)))
plt.legend()
plt.grid()
# plt.show()

# plot CG error profile
n_iter = len(J_cg_history)
x_error = np.arange(n_iter)
J_error = [J_stand] * n_iter
fig = plt.figure(title + 'CG Error Profile')
ax = fig.add_subplot(111)
ax.plot(x_error, J_error, 'r--', linewidth=2, label='Stand')
ax.plot(x_error, J_cg_history, 'g--', linewidth=2, label='CG')
plt.ylabel('Error Profile')
plt.xlabel('Iteration')
plt.title('CG Error Profile with M = ' + str(order) + ', N = ' + str(len(y_cg_approx)))
plt.legend()
plt.grid()


# plot computation time comparison
fig = plt.figure(title + 'Comparison Time (LOG)')
ax = fig.add_subplot(111)
cts = [np.log(ct_normal_eq), np.log(ct_grad_desc), np.log(ct_stoc_grad_desc), np.log(ct_stoc_grad_desc2), np.log(ct_cg)]
b = [0.15, 0.35, 0.55, 0.75, 0.95]
plt.xlim(0.0, 1.3)
tick_offset = [0.05] * 5
xticks = [x + y for x, y in zip(b, tick_offset)]
ax.set_xticks(xticks)
ax.set_xticklabels(('NE', 'GD', 'SGD', 'SGD2', 'CG'))
ax.bar(b, cts, width=0.1, color='r')
ax.set_yscale('symlog', linthreshy=1)
plt.xlabel('Methods')
plt.ylabel('Time (s)')
plt.title('Computation Time of NE,GD,SGD,SGD2,CG with Iter = ' + str(n_iter))
plt.grid()
# plt.show()


# plot computation time comparison
fig = plt.figure(title + 'Comparison Time')
ax = fig.add_subplot(111)
cts = [ct_normal_eq, ct_grad_desc, ct_stoc_grad_desc, ct_stoc_grad_desc2, ct_cg]
b = [0.15, 0.35, 0.55, 0.75, 0.95]
plt.xlim(0.0, 1.0)
tick_offset = [0.05] * 5
xticks = [x + y for x, y in zip(b, tick_offset)]
ax.set_xticks(xticks)
ax.set_xticklabels(('NE', 'GD', 'SGD', 'SGD2', 'CG'))
ax.bar(b, cts, width=0.1, color='r')
plt.xlabel('Methods')
plt.ylabel('Time (s)')
plt.title('Computation Time of NE,GD,SGD,SGD2,CG with Iter = ' + str(n_iter))
plt.grid()
plt.show()
