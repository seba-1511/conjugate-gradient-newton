#!/usr/bin/env python

import numpy as np
import matplotlib.pylab as plt
from numpy.linalg import norm, inv
from scipy.optimize import minimize_scalar

np.seterr(all='raise')

MAX_STEP = 10000


def conv_test(z_init, z_new):
    return norm(z_new - z_init) / max(1.0, norm(z_init))


def get_conv_color(res, epsilon=1e-3):
    res = np.rint(res)
    if norm(res - np.array([0, 1])) < epsilon:
        return 1
    if norm(res - np.array([0, -1])) < epsilon:
        return 2
    if norm(res - np.array([1, 0])) < epsilon:
        return 3
    if norm(res - np.array([-1, 0])) < epsilon:
        return 4
    return -2


def print_matrix(mat, name, interval):
    fig = plt.figure()
    cmap = plt.cm.get_cmap('PiYG', 11)
    plt.imshow(mat, cmap=cmap)
    # plt.colorbar()
    # plt.xlim([-interval, interval])
    # plt.ylim([-interval, interval])
    plt.savefig(name)


def gradient_descent(z_init, f, df, learning_rate=0.1, epsilon=1e-3):
    steps = 0
    while True:
        steps += 1
        z_new = z_init - (learning_rate * df(z_init))
        if conv_test(z_init, z_new) < epsilon or steps > MAX_STEP:
            return z_new, steps
        z_init = z_new


def newton_backtrack(z_init, f, df, Hf, epsilon=1e-3):
    steps = 0
    while True:
        steps += 1
        try:
            s = np.dot(inv(Hf(z_init)), df(z_init))
        except:
            return z_init, -1
        z_new = z_init - s
        # Backtracking step:
        while norm(df(z_new)) >= norm(df(z_init)):
            s /= 2.0
            z_new = z_init - s
        if conv_test(z_init, z_new) < epsilon or steps > MAX_STEP:
            return z_new, steps
        z_init = z_new


def conjugate_gradients(z_init, f, df, epsilon=1e-3):
    steps = 0
    d = -df(z_init)
    while True:
        steps += 1
        alpha = minimize_scalar(lambda a: f(z_init + a*d)).x
        z_new = z_init + alpha * d
        # Why convergence test needed ?
        if norm(df(z_new)) < epsilon or conv_test(z_init, z_new) < epsilon:
            return z_new, steps
        g_init = df(z_init)
        g_new = df(z_new)
        beta = (np.dot(g_new, g_new) - np.dot(g_new, g_init)) / np.dot(g_init, g_init)
        d = -df(z_new) + beta*d
        z_init = z_new


if __name__ == '__main__':
    f = lambda z: (z[0]**4 + z[1]**4 -6*(z[0]**2)*(z[1]**2) - 1)**2 + (4*(z[0]**3)*z[1] - 4*z[0]*(z[1]**3))**2
    df = lambda z: np.array([
        24*(z[0]**5)*(z[1]**2) + 24*(z[0]**3)*(z[1]**4) + 8*z[0]*(z[1]**6) + 8*(z[0]**7) - 8*(z[0]**3) + 24*z[0]*(z[1]**2),
        8*(z[0]**6)*z[1] + 24*(z[0]**4)*(z[1]**3) + 24*(z[0]**2)*(z[1]**5) + 8*(z[1]**7) - 8*(z[1]**3) + 24*(z[0]**2)*z[1],
    ], dtype=np.float64)
    Hf = lambda z: np.array([
        [
            120*(x**4)*(y**2) + 72*(x**2)*(y**4) + 8*(y**6) + 56*(x**6) - 24*(x**2) + 24*(y**2),
            48*(x**5)*y + 96*(x**3)*(y**3) + 48*x*(y**5) + 48*x*y,
        ],
        [
            48*(x**5)*y + 96*(x**3)*(y**3) + 48*x*(y**5) + 48*x*y,
            8*(x**6) + 72*(x**4)*(y**2) + 120*(x**2)*(y**4) + 56*(y**6) - 24*(y**2) + 24*(x**2),
        ],
    ])

    # f = lambda z: np.dot(z, z)
    # df = lambda z: np.array([
        # 2*z[0],
        # 2*z[1]
    # ])

    interval = 2.0
    resolution = 100.0

    desc_res = np.zeros((resolution, resolution))
    newton_res = np.zeros((resolution, resolution))
    conj_res = np.zeros((resolution, resolution))

    desc_steps = np.zeros((resolution, resolution))
    newton_steps = np.zeros((resolution, resolution))
    conj_steps = np.zeros((resolution, resolution))

    for k in xrange(int(resolution)):
        x = -interval + (2.0 * interval * k / resolution)
        for j in xrange(int(resolution)):
            y = -interval + (2.0 * interval * j / resolution)
            z = np.array([x, y])

            try:
                r_d, r_s = gradient_descent(z, f, df)
                print r_d
                r_d = get_conv_color(r_d)
            except:
                r_d = 0
                r_s = -1
            desc_res[k, j] = r_d
            desc_steps[k, j] = r_s

            try:
                n_d, n_s = newton_backtrack(z, f, df, Hf)
                n_d = get_conv_color(n_d)
            except:
                print 'exception'
                n_d = 0
                n_s = -1
            print n_d
            newton_res[k, j] = n_d
            newton_steps[k, j] = n_s

            try:
                c_d, c_s = conjugate_gradients(z, f, df)
                c_d = get_conv_color(c_d)
            except:
                c_d = 0
                c_s = -1
            conj_res[k, j] = c_d
            conj_steps[k, j] = c_s

    print_matrix(desc_steps, 'desc_steps.png', interval)
    print_matrix(desc_res, 'desc_results.png', interval)
    print_matrix(newton_steps, 'newton_steps.png', interval)
    print_matrix(newton_res, 'newton_results.png', interval)
    # print_matrix(conj_steps, 'conjugate_steps.png', interval)
    # print_matrix(conj_res, 'conjugate_results.png', interval)
