
import numpy as np
from time import time

# true signal
x = np.linspace(0.0, 2*np.pi, 100, True)
y_true = np.sin(x)

# noisy signal
n_sample = 10
x_sample = np.linspace(0.0, 2*np.pi, n_sample, True)
noise = np.random.normal(0, 0.15, len(x_sample))
y_sample = np.sin(x_sample) + noise

import grad_descent as reg
order = 3
n_gd_iter = 150
gd_alpha = 0.0001
sgd2_sample_size = 3
t0 = time()
w_stand, J_stand = reg.standReg(x_sample, y_sample, order)
t1 = time()
w_gd, J_gd_history = reg.gradDescent(x_sample, y_sample, order, n_gd_iter, gd_alpha)
print 'GD finished'
t2 = time()
w_sgd, J_sgd_history = reg.stocGradDescent(x_sample, y_sample, order, n_gd_iter, gd_alpha)
print 'SGD finished'
t3 = time()
w_sgd2, J_sgd2_history = reg.stocGradDescent2(x_sample, y_sample, order, n_gd_iter, gd_alpha, sgd2_sample_size)
print 'SGD2 finished'
t4 = time()
ct_normal_eq = t1 - t0
ct_grad_desc = t2 - t1
ct_stoc_grad_desc = t3 - t2
ct_stoc_grad_desc2 = t4 - t3

# create model for drawing
y_model_stand = reg.createModel(x, w_stand)
y_model_gd = reg.createModel(x, w_gd)
y_model_sgd = reg.createModel(x, w_sgd)
y_model_sgd2 = reg.createModel(x, w_sgd2)

# plot
import matplotlib.pyplot as plt

# plot results
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y_true, 'g-', linewidth=2, label='True')
ax.scatter(x_sample, y_sample, s=50, facecolors='none', edgecolors='b', linewidths=0.5, label='Data')
ax.plot(x, y_model_stand, 'r--', linewidth=2, label='Standard')
ax.plot(x, y_model_gd, 'g--', linewidth=2, label='GD')
ax.plot(x, y_model_sgd, 'm--', linewidth=2, label='SGD')
ax.plot(x, y_model_sgd2, 'c--', linewidth=2, label='SGD2')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression with M = ' + str(order) + ', N = ' + str(len(x_sample)))
plt.legend()
plt.grid()
# plt.show()

# plot GD error profile
n_iter = len(J_gd_history)
x_error = np.arange(n_iter)
J_error = [J_stand] * n_iter
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_error, J_error, 'r--', linewidth=2, label='Stand')
ax.plot(x_error, J_gd_history, 'g--', linewidth=2, label='GD')
plt.xlabel('Error Profile')
plt.ylabel('Iteration')
plt.title('GD Error Profile with M = ' + str(order) + ', N = ' + str(len(x_sample)))
plt.legend()
plt.grid()
# plt.show()

# plot SGD error profile
n_iter = len(J_sgd_history)
x_error = np.arange(n_iter)
J_error = [J_stand] * n_iter
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_error, J_error, 'r--', linewidth=2, label='Stand')
ax.plot(x_error, J_sgd_history, 'g--', linewidth=2, label='SGD')
plt.xlabel('Error Profile')
plt.ylabel('Iteration')
plt.title('SGD Error Profile with M = ' + str(order) + ', N = ' + str(len(x_sample)))
plt.legend()
plt.grid()
# plt.show()

# plot SGD2 error profile
n_iter = len(J_sgd2_history)
x_error = np.arange(n_iter)
J_error = [J_stand] * n_iter
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_error, J_error, 'r--', linewidth=2, label='Stand')
ax.plot(x_error, J_sgd2_history, 'g--', linewidth=2, label='SGD2')
plt.xlabel('Error Profile')
plt.ylabel('Iteration')
plt.title('SGD2 Error Profile with M = ' + str(order) + ', N = ' + str(len(x_sample)))
plt.legend()
plt.grid()
# plt.show()

# plot computation time comparison
fig = plt.figure()
ax = fig.add_subplot(111)
cts = [np.log(ct_normal_eq), np.log(ct_grad_desc), np.log(ct_stoc_grad_desc), np.log(ct_stoc_grad_desc2)]
b = [0.15, 0.35, 0.55, 0.75]
plt.xlim(0.0, 1.0)
tick_offset = [0.05] * 4
xticks = [x + y for x, y in zip(b, tick_offset)]
ax.set_xticks(xticks)
ax.set_xticklabels(('NE', 'GD', 'SGD', 'SGD2'))
ax.bar(b, cts, width=0.1, color='r')
ax.set_yscale('symlog', linthreshy=1)
plt.xlabel('Methods')
plt.ylabel('Time (s)')
plt.title('Computation Time of NE,GD,SGD,SGD2 with Iter = ' + str(n_iter))
plt.grid()
# plt.show()


# plot computation time comparison
fig = plt.figure()
ax = fig.add_subplot(111)
cts = [ct_normal_eq, ct_grad_desc, ct_stoc_grad_desc, ct_stoc_grad_desc2]
b = [0.15, 0.35, 0.55, 0.75]
plt.xlim(0.0, 1.0)
tick_offset = [0.05] * 4
xticks = [x + y for x, y in zip(b, tick_offset)]
ax.set_xticks(xticks)
ax.set_xticklabels(('NE', 'GD', 'SGD', 'SGD2'))
ax.bar(b, cts, width=0.1, color='r')
plt.xlabel('Methods')
plt.ylabel('Time (s)')
plt.title('Computation Time of NE,GD,SGD,SGD2 with Iter = ' + str(n_iter))
plt.grid()
plt.show()
