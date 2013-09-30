
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
n_gd_iter = 1500000
gd_alpha = 0.0001
sgd2_sample_size = 3

t0 = time()
w_anneal, J_anneal = reg.simulatedAnneal(x_sample, y_sample, order, num_iters=n_gd_iter)
print 'Annealing finished'
t1 = time()
w_gd, J_gd = reg.gradDescent(x_sample, y_sample, order, num_iters=n_gd_iter, alpha=gd_alpha)
print 'GD finished'
t2 = time()
w_newton, J_newton = reg.gradDescentNewton(x_sample, y_sample, order, num_iters=n_gd_iter)
print 'Newton finished'
t3 = time()



ct_anneal = t1 - t0
y_model_anneal = reg.createModel(x, w_anneal)
ct_gd = t2 - t1
y_model_gd = reg.createModel(x, w_gd)
ct_newton = t2 - t1
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
ax.plot(x, y_model_anneal, 'r--', linewidth=2, label='Anneal')
ax.plot(x, y_model_gd, 'g--', linewidth=2, label='GD')
ax.plot(x, y_model_newton, 'y--', linewidth=2, label='Newton')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression with M = ' + str(order) + ', N = ' + str(len(x_sample)))
plt.legend()
plt.grid()

# plot GD error profile
n_iter = len(J_gd)
x_error = np.arange(n_iter)
fig = plt.figure(title + 'GD Error Profile')
ax = fig.add_subplot(111)
ax.plot(x_error, J_anneal, 'r--', linewidth=2, label='Anneal')
ax.plot(x_error, J_gd, 'g--', linewidth=2, label='GD')
ax.plot(x_error, J_newton, 'y--', linewidth=2, label='Newton')
# ax.plot(x_error, J_cg_history, 'y--', linewidth=2, label='CG')
plt.ylabel('Error Profile')
plt.xlabel('Iteration')
plt.title('Error Profile with M = ' + str(order) + ', N = ' + str(len(x_sample)))
plt.legend()
plt.grid()

# plot computation time comparison in LOGARITHMIC
fig = plt.figure(title + 'Computation Time (LOG)')
ax = fig.add_subplot(111)
cts = [np.log(ct_anneal), np.log(ct_gd), np.log(ct_newton)]
b = [0.15, 0.35, 0.55]
plt.xlim(0.0, 0.8)
tick_offset = [0.05] * 3
xticks = [x + y for x, y in zip(b, tick_offset)]
ax.set_xticks(xticks)
ax.set_xticklabels(('Anneal', 'GD', 'Newton'))
ax.bar(b, cts, width=0.1, color='r')
ax.set_yscale('symlog', linthreshy=1)
plt.xlabel('Methods')
plt.ylabel('Time (s)')
plt.title('Computation Time of Annealing,GD,Newton with Iter = ' + str(n_gd_iter))
plt.grid()
plt.show()

# t0 = time()
# w_stand, J_stand = reg.standReg(x_sample, y_sample, order)
# t1 = time()
# w_gd, J_gd_history = reg.gradDescent(x_sample, y_sample, order, n_gd_iter, gd_alpha)
# print 'GD finished'
# t2 = time()
# w_sgd, J_sgd_history = reg.stocGradDescent(x_sample, y_sample, order, n_gd_iter, gd_alpha)
# print 'SGD finished'
# t3 = time()
# w_sgd2, J_sgd2_history = reg.stocGradDescent2(x_sample, y_sample, order, n_gd_iter, gd_alpha, sgd2_sample_size)
# print 'SGD2 finished'
# t4 = time()
# w_cg, J_cg_history = reg.conjugateGrad(x_sample, y_sample, order, n_gd_iter)
# print 'CG finished'
# t5 = time()

# ct_normal_eq = t1 - t0
# ct_grad_desc = t2 - t1
# ct_stoc_grad_desc = t3 - t2
# ct_stoc_grad_desc2 = t4 - t3
# ct_cg = t5 - t4

# # create model for drawing
# y_model_stand = reg.createModel(x, w_stand)
# y_model_gd = reg.createModel(x, w_gd)
# y_model_sgd = reg.createModel(x, w_sgd)
# y_model_sgd2 = reg.createModel(x, w_sgd2)
# y_model_cg = reg.createModel(x, w_cg)
#
# # plot
# import matplotlib.pyplot as plt
#
# # PLOT TITLE
# title = '[n' + str(n_sample) + '][m' + str(order) + '][iter' + str(n_gd_iter) + ']'
#
# # plot results
# fig = plt.figure(title + 'RESULT')
# ax = fig.add_subplot(111)
# ax.plot(x, y_true, 'g-', linewidth=2, label='True')
# ax.scatter(x_sample, y_sample, s=50, facecolors='none', edgecolors='b', linewidths=0.5, label='Data')
# ax.plot(x, y_model_stand, 'r--', linewidth=2, label='Standard')
# ax.plot(x, y_model_gd, 'g--', linewidth=2, label='GD')
# ax.plot(x, y_model_sgd, 'm--', linewidth=2, label='SGD')
# ax.plot(x, y_model_sgd2, 'c--', linewidth=2, label='SGD2')
# ax.plot(x, y_model_cg, 'y--', linewidth=2, label='CG')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('Regression with M = ' + str(order) + ', N = ' + str(len(x_sample)))
# plt.legend()
# plt.grid()
# # plt.show()
#
# # plot GD error profile
# n_iter = len(J_gd_history)
# x_error = np.arange(n_iter)
# J_error = [J_stand] * n_iter
# fig = plt.figure(title + 'GD Error Profile')
# ax = fig.add_subplot(111)
# ax.plot(x_error, J_error, 'r--', linewidth=2, label='Stand')
# ax.plot(x_error, J_gd_history, 'g--', linewidth=2, label='GD')
# # ax.plot(x_error, J_cg_history, 'y--', linewidth=2, label='CG')
# plt.ylabel('Error Profile')
# plt.xlabel('Iteration')
# plt.title('GD Error Profile with M = ' + str(order) + ', N = ' + str(len(x_sample)))
# plt.legend()
# plt.grid()
# # plt.show()
#
# # plot SGD error profile
# n_iter = len(J_sgd_history)
# x_error = np.arange(n_iter)
# J_error = [J_stand] * n_iter
# fig = plt.figure(title + 'SGD Error Profile')
# ax = fig.add_subplot(111)
# ax.plot(x_error, J_error, 'r--', linewidth=2, label='Stand')
# ax.plot(x_error, J_sgd_history, 'g--', linewidth=2, label='SGD')
# plt.ylabel('Error Profile')
# plt.xlabel('Iteration')
# plt.title('SGD Error Profile with M = ' + str(order) + ', N = ' + str(len(x_sample)))
# plt.legend()
# plt.grid()
# # plt.show()
#
# # plot SGD2 error profile
# n_iter = len(J_sgd2_history)
# x_error = np.arange(n_iter)
# J_error = [J_stand] * n_iter
# fig = plt.figure(title + 'SGD2 Error Profile')
# ax = fig.add_subplot(111)
# ax.plot(x_error, J_error, 'r--', linewidth=2, label='Stand')
# ax.plot(x_error, J_sgd2_history, 'g--', linewidth=2, label='SGD2')
# plt.ylabel('Error Profile')
# plt.xlabel('Iteration')
# plt.title('SGD2 Error Profile with M = ' + str(order) + ', N = ' + str(len(x_sample)))
# plt.legend()
# plt.grid()
# # plt.show()
#
# # plot CG error profile
# n_iter = len(J_cg_history)
# x_error = np.arange(n_iter)
# J_error = [J_stand] * n_iter
# fig = plt.figure(title + 'CG Error Profile')
# ax = fig.add_subplot(111)
# ax.plot(x_error, J_error, 'r--', linewidth=2, label='Stand')
# ax.plot(x_error, J_cg_history, 'g--', linewidth=2, label='CG')
# plt.ylabel('Error Profile')
# plt.xlabel('Iteration')
# plt.title('CG Error Profile with M = ' + str(order) + ', N = ' + str(len(x_sample)))
# plt.legend()
# plt.grid()
#
# # plot computation time comparison in LOGARITHMIC
# fig = plt.figure(title + 'Computation Time (LOG)')
# ax = fig.add_subplot(111)
# cts = [np.log(ct_normal_eq), np.log(ct_grad_desc), np.log(ct_stoc_grad_desc), np.log(ct_stoc_grad_desc2), np.log(ct_cg)]
# b = [0.15, 0.35, 0.55, 0.75, 0.95]
# plt.xlim(0.0, 1.3)
# tick_offset = [0.05] * 5
# xticks = [x + y for x, y in zip(b, tick_offset)]
# ax.set_xticks(xticks)
# ax.set_xticklabels(('NE', 'GD', 'SGD', 'SGD2', 'CG'))
# ax.bar(b, cts, width=0.1, color='r')
# ax.set_yscale('symlog', linthreshy=1)
# plt.xlabel('Methods')
# plt.ylabel('Time (s)')
# plt.title('Computation Time of NE,GD,SGD,SGD2,CG with Iter = ' + str(n_iter))
# plt.grid()
# # plt.show()
#
#
# # plot computation time comparison
# fig = plt.figure(title + 'Computation Time')
# ax = fig.add_subplot(111)
# cts = [ct_normal_eq, ct_grad_desc, ct_stoc_grad_desc, ct_stoc_grad_desc2, ct_cg]
# b = [0.15, 0.35, 0.55, 0.75, 0.95]
# plt.xlim(0.0, 1.3)
# tick_offset = [0.05] * 5
# xticks = [x + y for x, y in zip(b, tick_offset)]
# ax.set_xticks(xticks)
# ax.set_xticklabels(('NE', 'GD', 'SGD', 'SGD2', 'CG'))
# ax.bar(b, cts, width=0.1, color='r')
# plt.xlabel('Methods')
# plt.ylabel('Time (s)')
# plt.title('Computation Time of NE,GD,SGD,SGD2.CG with Iter = ' + str(n_iter))
# plt.grid()
# plt.show()
