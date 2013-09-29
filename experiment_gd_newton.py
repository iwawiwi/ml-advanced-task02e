
import numpy as np
from time import time

# true signal
x = np.linspace(0.0, 2*np.pi, 100, True)
y_true = np.sin(x)

# noisy signal
n_sample = 10000
x_sample = np.linspace(0.0, 2*np.pi, n_sample, True)
noise = np.random.normal(0, 0.15, len(x_sample))
y_sample = np.sin(x_sample) + noise

import grad_descent as reg
order = 3
n_gd_iter = 15000
gd_alpha = 0.0001
sgd2_sample_size = 3
t0 = time()
w_gd, J_gd_history = reg.gradDescent(x_sample, y_sample, order, n_gd_iter, gd_alpha)
print 'GD finished'
t1 = time()
w_cg, J_cg_history = reg.conjugateGrad(x_sample, y_sample, order, n_gd_iter)
print 'CG finished'
t2 = time()
w_newton, J_newton_history = reg.gradDescentNewton(x_sample, y_sample, order, n_gd_iter)
print 'Newton finished'
t3 = time()


ct_grad_desc = t1 - t0
ct_cg = t2 - t1
ct_newton = t3 - t2

# create model for drawing
y_model_gd = reg.createModel(x, w_gd)
y_model_cg = reg.createModel(x, w_cg)
y_model_newton = reg.createModel(x, w_newton)

# plot
import matplotlib.pyplot as plt

# PLOT TITLE
title = '[n' + str(n_sample) + '][m' + str(order) + '][iter' + str(n_gd_iter) + ']'

# plot results
fig = plt.figure(title + 'RESULT')
ax = fig.add_subplot(111)
ax.plot(x, y_true, 'g-', linewidth=2, label='True')
ax.scatter(x_sample, y_sample, s=50, facecolors='none', edgecolors='b', linewidths=0.5, label='Data')
ax.plot(x, y_model_gd, 'g--', linewidth=2, label='GD')
ax.plot(x, y_model_cg, 'y--', linewidth=2, label='CG')
ax.plot(x, y_model_newton, 'r--', linewidth=2, label='Newton')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression with M = ' + str(order) + ', N = ' + str(len(x_sample)))
plt.legend()
plt.grid()
# plt.show()

# plot Error Profile
n_iter = len(J_gd_history)
x_error = np.arange(n_iter)
fig = plt.figure(title + 'GD Error Profile')
ax = fig.add_subplot(111)
ax.plot(x_error, J_gd_history, 'g--', linewidth=2, label='GD')
ax.plot(x_error, J_cg_history, 'y--', linewidth=2, label='CG')
ax.plot(x_error, J_newton_history, 'r--', linewidth=2, label='Newton')
plt.ylabel('Error Profile')
plt.xlabel('Iteration')
plt.title('Error Profile with M = ' + str(order) + ', N = ' + str(len(x_sample)))
plt.legend()
plt.grid()
# plt.show()

# plot computation time comparison in LOGARITHMIC
fig = plt.figure(title + 'Computation Time (LOG)')
ax = fig.add_subplot(111)
cts = [np.log(ct_grad_desc), np.log(ct_cg), np.log(ct_newton)]
b = [0.15, 0.35, 0.55]
plt.xlim(0.0, 0.8)
tick_offset = [0.05] * 3
xticks = [x + y for x, y in zip(b, tick_offset)]
ax.set_xticks(xticks)
ax.set_xticklabels(('GD', 'CG', 'Newton'))
ax.bar(b, cts, width=0.1, color='r')
ax.set_yscale('symlog', linthreshy=1)
plt.xlabel('Methods')
plt.ylabel('Time (s)')
plt.title('Computation Time of GD,CG,Newton with Iter = ' + str(n_iter))
plt.grid()
# plt.show()

plt.show()
